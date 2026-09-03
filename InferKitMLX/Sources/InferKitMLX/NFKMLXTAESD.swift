//
//  NFKMLXTAESD.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// TAESD (Tiny AutoEncoder for Stable Diffusion): a small distilled autoencoder that maps an image to a
// four-channel latent and back, the fast preview decoder for a latent-diffusion pipeline. The encoder
// and decoder are flat `nn.Sequential` stacks of 3×3 convolutions and residual blocks; modeling them as
// `[Module]` arrays makes the numeric Sequential keys (`0.weight`, `1.conv.0.weight`, …) match with no
// remap. Tensors flow as `[batch, H, W, channels]` (MLX's NHWC).

/// A parameter-free op occupying a Sequential index: ReLU, the decoder's input clamp `tanh(x/3)·3`, or a
/// nearest ×2 upsample. Kept as a `Module` so the checkpoint's numeric indices line up.
final class NFKTAESDReLU: Module {}
final class NFKTAESDClamp: Module {}
final class NFKTAESDUpsample: Module {}

/// A TAESD residual block: three 3×3 convolutions (ReLU between them), added back to the input, then a
/// fusing ReLU. Every block here is 64→64, so the skip is the identity.
final class NFKTAESDBlock: Module {
    @ModuleInfo(key: "conv") var conv: [Module]                          // [Conv2d, ReLU, Conv2d, ReLU, Conv2d]

    override init() {
        func c() -> Conv2d { Conv2d(inputChannels: 64, outputChannels: 64, kernelSize: 3, padding: 1) }
        _conv.wrappedValue = [c(), NFKTAESDReLU(), c(), NFKTAESDReLU(), c()]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in conv {
            if let convolution = layer as? Conv2d { h = convolution(h) } else { h = relu(h) }
        }
        return relu(h + x)
    }
}

/// Runs a TAESD Sequential (`[Module]` of convolutions, blocks, and the parameter-free ops).
private func runTAESDSequence(_ layers: [Module], _ input: MLXArray) -> MLXArray {
    var x = input
    for layer in layers {
        switch layer {
        case let convolution as Conv2d: x = convolution(x)
        case let block as NFKTAESDBlock: x = block(x)
        case is NFKTAESDReLU: x = relu(x)
        case is NFKTAESDClamp: x = tanh(x / 3) * 3
        case is NFKTAESDUpsample: x = NFKMLXResample.upsampleNearest(x, scale: 2)
        default: break
        }
    }
    return x
}

/// The TAESD network: an 8× downsampling encoder to a `latentChannels`-wide latent and an 8× upsampling
/// decoder back to RGB.
final class NFKMLXTAESDNet: Module {
    @ModuleInfo(key: "encoder") var encoder: [Module]
    @ModuleInfo(key: "decoder") var decoder: [Module]

    let latentChannels: Int

    init(latentChannels: Int = 4) {
        self.latentChannels = latentChannels
        func conv(_ inCh: Int, _ outCh: Int, stride: Int = 1, bias: Bool = true) -> Conv2d {
            Conv2d(inputChannels: inCh, outputChannels: outCh, kernelSize: 3, stride: IntOrPair(stride),
                   padding: 1, bias: bias)
        }
        func blocks(_ n: Int) -> [Module] { (0 ..< n).map { _ in NFKTAESDBlock() } }
        // A downsample is a stride-2, bias-free convolution; each is followed by three blocks.
        _encoder.wrappedValue = [conv(3, 64), NFKTAESDBlock()]
            + [conv(64, 64, stride: 2, bias: false)] + blocks(3)
            + [conv(64, 64, stride: 2, bias: false)] + blocks(3)
            + [conv(64, 64, stride: 2, bias: false)] + blocks(3)
            + [conv(64, latentChannels)]
        // The decoder clamps, projects in, then upsamples (nearest ×2 + bias-free conv) between blocks.
        _decoder.wrappedValue = [NFKTAESDClamp(), conv(latentChannels, 64), NFKTAESDReLU()]
            + blocks(3) + [NFKTAESDUpsample(), conv(64, 64, bias: false)]
            + blocks(3) + [NFKTAESDUpsample(), conv(64, 64, bias: false)]
            + blocks(3) + [NFKTAESDUpsample(), conv(64, 64, bias: false)]
            + [NFKTAESDBlock(), conv(64, 3)]
    }

    /// Image `[B, H, W, 3]` in `0…1` → latent `[B, H/8, W/8, latentChannels]`.
    func encode(_ image: MLXArray) -> MLXArray { runTAESDSequence(encoder, image) }

    /// Latent `[B, h, w, latentChannels]` → image `[B, h·8, w·8, 3]`.
    func decode(_ latent: MLXArray) -> MLXArray { runTAESDSequence(decoder, latent) }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKTAESDHolder: @unchecked Sendable {
    let net: NFKMLXTAESDNet
    init(_ net: NFKMLXTAESDNet) { self.net = net }
}

/// TAESD reconstruction (image → latent → image) as an InferKit backend, reading `NFKInputImage` and
/// returning the reconstruction under `NFKOutputImage`. The latent itself is reached through
/// `NFKMLXTAESD.encode`, which is the fast preview decoder a diffusion pipeline uses.
@objc(NFKMLXTAESDBackend)
public final class NFKMLXTAESDBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKTAESDHolder
    private let identifier: String

    init(net: NFKMLXTAESDNet, identifier: String) {
        holder = NFKTAESDHolder(net)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let value = request.input(forKey: NFKInputImage) else { throw NFKMLXError.unsupportedInput }
        let image = try NFKMLXImageBridge.tensor(from: value, channels: 3, colorSpace: CGColorSpaceCreateDeviceRGB())
        let batched = image.reshaped([1, image.shape[0], image.shape[1], 3])
        let reconstruction = holder.net.decode(holder.net.encode(batched))
        eval(reconstruction)
        let output = try NFKMLXImageBridge.cgImage(from: reconstruction[0], options: NFKMLXImageOptions())
        return NFKInferenceResult(outputs: [NFKOutputImage: output])
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        Task.detached(priority: .userInitiated) {
            do {
                job.finish(with: try self.runInference(for: request))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }
}

/// Registration, encode/decode, and weight loading for TAESD.
@objc(NFKMLXTAESD)
public final class NFKMLXTAESD: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "taesd"

    private let holder: NFKTAESDHolder

    init(net: NFKMLXTAESDNet) { holder = NFKTAESDHolder(net) }

    static func makeNet(latentChannels: Int = 4) -> NFKMLXTAESDNet { NFKMLXTAESDNet(latentChannels: latentChannels) }

    /// Encodes an image (`[H, W, 3]` in `0…1`, as an MLXArray) to its latent `[H/8, W/8, latentChannels]`.
    public func encode(_ image: MLXArray) -> MLXArray {
        let batched = image.ndim == 3 ? image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]]) : image
        let latent = holder.net.encode(batched)
        eval(latent)
        return latent
    }

    /// Decodes a latent to an image in `0…1`.
    public func decode(_ latent: MLXArray) -> MLXArray {
        let image = holder.net.decode(latent.ndim == 3 ? latent.reshaped([1, latent.shape[0], latent.shape[1], latent.shape[2]]) : latent)
        eval(image)
        return image
    }

    /// Builds a TAESD backend directly from optional local weights.
    ///
    /// - Since: InferKit 0.3.1
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        NFKMLXTAESDBackend(net: try loadedNet(weightsURL: weightsURL), identifier: modelName)
    }

    /// Builds a TAESD codec object (with `encode`/`decode`) from optional local weights.
    public static func codec(weightsURL: URL?) throws -> NFKMLXTAESD {
        NFKMLXTAESD(net: try loadedNet(weightsURL: weightsURL))
    }

    private static func loadedNet(weightsURL: URL?) throws -> NFKMLXTAESDNet {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// Downloads the checkpoint from Hugging Face, then builds.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers TAESD (`taesd`) with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a combined checkpoint (`encoder.N.*` / `decoder.N.*` in PyTorch layout), transposing the
    /// 4-D Conv2d weights `[out, in, kH, kW]` → MLX's `[out, kH, kW, in]`. The `[Module]`-array layout
    /// makes the numeric Sequential keys match directly, so there is no name remap.
    static func loadWeights(into net: NFKMLXTAESDNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let transpose = checkpoint.needsConvTranspose
        let mapped = checkpoint.arrays.map { key, value -> (String, MLXArray) in
            (transpose && value.ndim == 4) ? (key, value.transposed(0, 2, 3, 1)) : (key, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
