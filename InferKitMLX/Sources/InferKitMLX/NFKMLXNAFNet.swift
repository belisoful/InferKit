//
//  NFKMLXNAFNet.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// NAFNet (Nonlinear Activation Free Network) is a single-forward image restoration network (denoise,
// deblur). It runs through `NFKMLXModuleBackend`: a degraded image in, the restored image out, added
// to the input as a global residual. Tensors flow NHWC, so NAFNet's channel-wise LayerNorm2d is just
// a LayerNorm over the last axis.
//
// Block-internal names (`conv1`…`conv5`, `norm1`, `norm2`, `beta`, `gamma`) match the reference; the
// channel-attention conv (`sca`) and the encoder/decoder/middle lists differ from the reference
// nesting, so mapping a real checkpoint is a `remap` (validation-sweep task).

/// How NAFNet is sized. Defaults are the SIDD denoiser (width 32).
public struct NFKMLXNAFNetConfiguration: Sendable {
    public var width: Int = 32
    public var encoderBlocks: [Int] = [2, 2, 4, 8]
    public var middleBlocks: Int = 12
    public var decoderBlocks: [Int] = [2, 2, 2, 2]
    public init() {}

    /// The released SIDD denoiser: blocks spread through the middle of the stack.
    public static let sidd = NFKMLXNAFNetConfiguration()

    /// The released GoPro deblurrer: the same width, but twenty-eight of its blocks sit in the last
    /// encoder stage and only one in the middle.
    public static var goPro: NFKMLXNAFNetConfiguration {
        var configuration = NFKMLXNAFNetConfiguration()
        configuration.encoderBlocks = [1, 1, 1, 28]
        configuration.middleBlocks = 1
        configuration.decoderBlocks = [1, 1, 1, 1]
        return configuration
    }

    /// The released REDS model: the GoPro block distribution at twice the width.
    public static var reds: NFKMLXNAFNetConfiguration {
        var configuration = NFKMLXNAFNetConfiguration.goPro
        configuration.width = 64
        return configuration
    }

    var levels: Int { encoderBlocks.count }
}

/// A NAFBlock: SimpleGate + Simplified Channel Attention, then a gated feed-forward, each a residual.
final class NFKNAFBlock: Module {
    let norm1: LayerNorm
    let conv1: Conv2d
    let conv2: Conv2d                                           // depthwise
    let conv3: Conv2d
    let sca: Conv2d
    let norm2: LayerNorm
    let conv4: Conv2d
    let conv5: Conv2d
    let beta: MLXArray
    let gamma: MLXArray

    init(channels c: Int) {
        norm1 = LayerNorm(dimensions: c)
        conv1 = Conv2d(inputChannels: c, outputChannels: c * 2, kernelSize: 1)
        conv2 = Conv2d(inputChannels: c * 2, outputChannels: c * 2, kernelSize: 3, padding: 1, groups: c * 2)
        conv3 = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1)
        sca = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1)
        norm2 = LayerNorm(dimensions: c)
        conv4 = Conv2d(inputChannels: c, outputChannels: c * 2, kernelSize: 1)
        conv5 = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1)
        beta = MLXArray.zeros([1, 1, 1, c])
        gamma = MLXArray.zeros([1, 1, 1, c])
    }

    static func simpleGate(_ x: MLXArray) -> MLXArray {
        let parts = x.split(parts: 2, axis: 3)
        return parts[0] * parts[1]
    }

    func callAsFunction(_ inp: MLXArray) -> MLXArray {
        var x = conv1(norm1(inp))
        x = conv2(x)
        x = Self.simpleGate(x)
        x = x * sca(x.mean(axes: [1, 2], keepDims: true))       // Simplified Channel Attention
        x = conv3(x)
        let y = inp + x * beta

        var z = conv4(norm2(y))
        z = Self.simpleGate(z)
        z = conv5(z)
        return y + z * gamma
    }
}

/// The NAFNet U-shaped restoration network. Input `[1, H, W, 3]` → restored `[1, H, W, 3]`.
final class NFKMLXNAFNetNet: Module {
    @ModuleInfo(key: "intro") var intro: Conv2d
    @ModuleInfo(key: "encoders") var encoders: [[NFKNAFBlock]]
    @ModuleInfo(key: "downs") var downs: [Conv2d]
    @ModuleInfo(key: "middle") var middle: [NFKNAFBlock]
    @ModuleInfo(key: "ups") var ups: [Conv2d]
    @ModuleInfo(key: "decoders") var decoders: [[NFKNAFBlock]]
    @ModuleInfo(key: "ending") var ending: Conv2d

    let levels: Int

    init(_ configuration: NFKMLXNAFNetConfiguration) {
        levels = configuration.levels
        _intro.wrappedValue = Conv2d(inputChannels: 3, outputChannels: configuration.width, kernelSize: 3, padding: 1)

        var channels = configuration.width
        var encoderStages = [[NFKNAFBlock]]()
        var downConvs = [Conv2d]()
        for count in configuration.encoderBlocks {
            encoderStages.append((0 ..< count).map { _ in NFKNAFBlock(channels: channels) })
            downConvs.append(Conv2d(inputChannels: channels, outputChannels: channels * 2, kernelSize: 2, stride: 2))
            channels *= 2
        }
        _encoders.wrappedValue = encoderStages
        _downs.wrappedValue = downConvs
        _middle.wrappedValue = (0 ..< configuration.middleBlocks).map { _ in NFKNAFBlock(channels: channels) }

        var upConvs = [Conv2d]()
        var decoderStages = [[NFKNAFBlock]]()
        for count in configuration.decoderBlocks {
            upConvs.append(Conv2d(inputChannels: channels, outputChannels: channels * 2, kernelSize: 1, bias: false))
            channels /= 2
            decoderStages.append((0 ..< count).map { _ in NFKNAFBlock(channels: channels) })
        }
        _ups.wrappedValue = upConvs
        _decoders.wrappedValue = decoderStages
        _ending.wrappedValue = Conv2d(inputChannels: configuration.width, outputChannels: 3, kernelSize: 3, padding: 1)
    }

    /// Channel-major PixelShuffle ×2: `[N, H, W, C]` → `[N, 2H, 2W, C/4]`.
    static func pixelShuffle2(_ x: MLXArray) -> MLXArray {
        let (n, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        return x.reshaped([n, h, w, c / 4, 2, 2])
            .transposed(0, 1, 4, 2, 5, 3)
            .reshaped([n, h * 2, w * 2, c / 4])
    }

    private func run(_ input: MLXArray) -> MLXArray {
        var x = intro(input)
        var skips = [MLXArray]()
        for (stage, down) in zip(encoders, downs) {
            for block in stage { x = block(x) }
            skips.append(x)
            x = down(x)
        }
        for block in middle { x = block(x) }
        for (index, (up, stage)) in zip(ups, decoders).enumerated() {
            x = Self.pixelShuffle2(up(x))
            x = x + skips[skips.count - 1 - index]
            for block in stage { x = block(x) }
        }
        return ending(x) + input
    }

    /// Restores a bridged image `[H, W, 3]` (`0...1`). Pads to a multiple of `2^levels` for the
    /// symmetric down/up path, then crops back.
    func restore(_ image: MLXArray) -> MLXArray {
        let (height, width) = (image.shape[0], image.shape[1])
        let multiple = 1 << levels
        let padH = (multiple - height % multiple) % multiple
        let padW = (multiple - width % multiple) % multiple
        var input = image.reshaped([1, height, width, image.shape[2]])
        if padH > 0 || padW > 0 {
            input = padded(input, widths: [IntOrPair((0, 0)), IntOrPair((0, padH)), IntOrPair((0, padW)), IntOrPair((0, 0))], mode: .edge)
        }
        let output = clip(run(input), min: 0, max: 1)
        return output[0, 0 ..< height, 0 ..< width, 0...]
    }
}

/// NAFNet image restoration as an InferKit backend, and its registration for the Objective-C path.
/// Which released NAFNet a backend is built for.
@objc(NFKMLXNAFNetVariant)
public enum NFKMLXNAFNetVariant: Int {
    /// The SIDD denoiser.
    case sidd
    /// The GoPro deblurrer.
    case goPro
    /// The REDS model: the GoPro distribution at twice the width.
    case reds
}

@objc(NFKMLXNAFNet)
public final class NFKMLXNAFNet: NSObject {

    @objc public static let modelName = "nafnet"

    static func makeNet(_ configuration: NFKMLXNAFNetConfiguration = NFKMLXNAFNetConfiguration()) -> NFKMLXNAFNetNet {
        NFKMLXNAFNetNet(configuration)
    }

    /// Builds a NAFNet restoration backend directly from optional local weights — no registry required.
    /// A nil `weightsURL` builds random weights (`isReady` is true).
    /// Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        try backend(variant: .sidd, weightsURL: weightsURL)
    }

    /// Builds at one of the released geometries. The two differ in how their blocks are distributed,
    /// so a checkpoint only loads against the one it was trained as.
    @objc(backendWithVariant:weightsURL:error:)
    public static func backend(variant: NFKMLXNAFNetVariant, weightsURL: URL?) throws -> any NFKInferenceBackend {
        let geometry: NFKMLXNAFNetConfiguration
        switch variant {
        case .sidd: geometry = .sidd
        case .goPro: geometry = .goPro
        case .reds: geometry = .reds
        }
        let net = NFKMLXNAFNetNet(geometry)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXNAFNetHolder(net)
        return NFKMLXModuleBackend(identifier: modelName, isReady: true) { image in holder.net.restore(image) }
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required.
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

    /// Registers NAFNet (`nafnet`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a safetensors checkpoint, transposing 4-D convolution weights to MLX's layout. `remap`
    /// maps the reference `sca.1`/`encoders.N.M` names to the module's keys (validation-sweep task).
    static func loadWeights(into net: NFKMLXNAFNetNet, from url: URL, remap: (String) -> String = { $0 }) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value in
            (remap(key), checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}

private final class NFKMLXNAFNetHolder: @unchecked Sendable {
    let net: NFKMLXNAFNetNet
    init(_ net: NFKMLXNAFNetNet) { self.net = net }
}
