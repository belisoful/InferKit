//
//  NFKMLXConvTasNet.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// Conv-TasNet separates a mono mixture into several speakers directly in the time domain. A 1-D
// convolutional encoder turns the waveform into a non-negative feature representation; a temporal
// convolutional network (a stack of depthwise-separable dilated 1-D convolutions with global layer
// normalization) predicts one multiplicative mask per speaker; the masked representations are inverted
// by a shared transposed-convolution decoder back to waveforms.
//
// Module structure follows the reference Conv-TasNet. Names are grouped cleanly (`encoder`, `blocks.N`,
// `decoder`) rather than the reference's `separator.network.*`, which is what `remapReferenceKey`
// translates. Every PReLU carries ONE learnable slope, which is the reference's own shape: asteroid
// builds them as `nn.PReLU()`, whose `num_parameters` defaults to 1, and all 49 slope tensors in the
// released checkpoint are `[1]`. Tensors flow as `[batch, time, channels]` (MLX 1-D convolution
// layout).

/// Conv-TasNet dimensions. Defaults follow the reference; `tiny` keeps tests fast.
public struct NFKMLXConvTasNetConfiguration: Sendable {
    public var filters: Int          // encoder basis size (N)
    public var kernel: Int           // encoder/decoder kernel (L); stride is L/2
    public var bottleneck: Int       // TCN bottleneck channels (B)
    public var hidden: Int           // TCN hidden channels (H)
    public var convKernel: Int       // depthwise conv kernel (P)
    public var blocksPerRepeat: Int  // X
    public var repeats: Int          // R
    public var speakers: Int         // C

    /// Whether each PReLU learns one slope per channel instead of a single shared slope.
    ///
    /// The released checkpoint carries one slope per PReLU, because asteroid builds them as
    /// `nn.PReLU()` and `num_parameters` defaults to 1. That is the default here, so a released
    /// checkpoint loads into the shape it was trained as.
    ///
    /// Set this to give a fine-tune more capacity in the activations. A shared slope broadcast across
    /// channels is the same function, so a released checkpoint still loads and computes identically —
    /// `loadWeights` widens its `[1]` slopes to `[C]`, and training moves them apart from there.
    public var perChannelPReLU: Bool

    public init(filters: Int = 512, kernel: Int = 16, bottleneck: Int = 128, hidden: Int = 512,
                convKernel: Int = 3, blocksPerRepeat: Int = 8, repeats: Int = 3, speakers: Int = 2,
                perChannelPReLU: Bool = false) {
        self.perChannelPReLU = perChannelPReLU
        self.filters = filters
        self.kernel = kernel
        self.bottleneck = bottleneck
        self.hidden = hidden
        self.convKernel = convKernel
        self.blocksPerRepeat = blocksPerRepeat
        self.repeats = repeats
        self.speakers = speakers
    }

    public static let base = NFKMLXConvTasNetConfiguration()

    /// The released `JorisCos/ConvTasNet_Libri2Mix_sepclean_16k` geometry (a 32-sample filterbank).
    public static let libri2Mix16k = NFKMLXConvTasNetConfiguration(filters: 512, kernel: 32, bottleneck: 128,
                                                                   hidden: 512, convKernel: 3,
                                                                   blocksPerRepeat: 8, repeats: 3, speakers: 2)

    public static let tiny = NFKMLXConvTasNetConfiguration(filters: 32, kernel: 8, bottleneck: 16, hidden: 32,
                                                           convKernel: 3, blocksPerRepeat: 2, repeats: 1, speakers: 2)

    var speakerNames: [String] { (1 ... speakers).map { "speaker-\($0)" } }
}

/// Global layer normalization: normalizes over time and channels together, with a per-channel affine.
final class NFKTasNetGlobalNorm: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray

    init(channels: Int) {
        _weight.wrappedValue = MLXArray.ones([1, 1, channels])
        _bias.wrappedValue = MLXArray.zeros([1, 1, channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axes: [1, 2], keepDims: true)
        let variance = ((x - mean) * (x - mean)).mean(axes: [1, 2], keepDims: true)
        return (x - mean) * rsqrt(variance + 1e-8) * weight + bias
    }
}

/// One temporal block: a pointwise expansion, a depthwise dilated convolution, and residual/skip
/// projections, each nonlinearity preceded by global layer norm.
final class NFKTasNetBlock: Module {
    @ModuleInfo(key: "conv1x1") var conv1x1: Conv1d
    @ModuleInfo(key: "prelu1") var prelu1: PReLU
    @ModuleInfo(key: "norm1") var norm1: NFKTasNetGlobalNorm
    @ModuleInfo(key: "dconv") var dconv: Conv1d
    @ModuleInfo(key: "prelu2") var prelu2: PReLU
    @ModuleInfo(key: "norm2") var norm2: NFKTasNetGlobalNorm
    @ModuleInfo(key: "residual") var residual: Conv1d
    @ModuleInfo(key: "skip") var skip: Conv1d

    init(bottleneck: Int, hidden: Int, kernel: Int, dilation: Int, perChannel: Bool = false) {
        _conv1x1.wrappedValue = Conv1d(inputChannels: bottleneck, outputChannels: hidden, kernelSize: 1)
        _prelu1.wrappedValue = PReLU(count: perChannel ? hidden : 1)
        _norm1.wrappedValue = NFKTasNetGlobalNorm(channels: hidden)
        _dconv.wrappedValue = Conv1d(inputChannels: hidden, outputChannels: hidden, kernelSize: kernel,
                                     padding: dilation * (kernel - 1) / 2, dilation: dilation, groups: hidden)
        _prelu2.wrappedValue = PReLU(count: perChannel ? hidden : 1)
        _norm2.wrappedValue = NFKTasNetGlobalNorm(channels: hidden)
        _residual.wrappedValue = Conv1d(inputChannels: hidden, outputChannels: bottleneck, kernelSize: 1)
        _skip.wrappedValue = Conv1d(inputChannels: hidden, outputChannels: bottleneck, kernelSize: 1)
    }

    /// Returns the residual sum (fed to the next block) and the skip contribution (accumulated).
    func callAsFunction(_ x: MLXArray) -> (residual: MLXArray, skip: MLXArray) {
        var y = norm1(prelu1(conv1x1(x)))
        y = norm2(prelu2(dconv(y)))
        return (x + residual(y), skip(y))
    }
}

/// The Conv-TasNet network: encoder, masking TCN, and a shared decoder.
final class NFKMLXConvTasNetNet: Module {
    @ModuleInfo(key: "encoder") var encoder: Conv1d
    @ModuleInfo(key: "input_norm") var inputNorm: NFKTasNetGlobalNorm
    @ModuleInfo(key: "bottleneck") var bottleneck: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [NFKTasNetBlock]
    @ModuleInfo(key: "mask_prelu") var maskPReLU: PReLU
    @ModuleInfo(key: "mask_conv") var maskConv: Conv1d
    @ModuleInfo(key: "decoder") var decoder: NFKDemucsConvT1d

    let configuration: NFKMLXConvTasNetConfiguration

    init(_ c: NFKMLXConvTasNetConfiguration) {
        configuration = c
        let stride = c.kernel / 2
        // The reference's encoder and decoder are filterbanks: a weight matrix with no bias.
        _encoder.wrappedValue = Conv1d(inputChannels: 1, outputChannels: c.filters, kernelSize: c.kernel,
                                       stride: stride, bias: false)
        _inputNorm.wrappedValue = NFKTasNetGlobalNorm(channels: c.filters)
        _bottleneck.wrappedValue = Conv1d(inputChannels: c.filters, outputChannels: c.bottleneck, kernelSize: 1)
        var tcn: [NFKTasNetBlock] = []
        for _ in 0 ..< c.repeats {
            for x in 0 ..< c.blocksPerRepeat {
                tcn.append(NFKTasNetBlock(bottleneck: c.bottleneck, hidden: c.hidden,
                                          kernel: c.convKernel, dilation: 1 << x,
                                          perChannel: c.perChannelPReLU))
            }
        }
        _blocks.wrappedValue = tcn
        _maskPReLU.wrappedValue = PReLU(count: c.perChannelPReLU ? c.bottleneck : 1)
        _maskConv.wrappedValue = Conv1d(inputChannels: c.bottleneck, outputChannels: c.filters * c.speakers, kernelSize: 1)
        _decoder.wrappedValue = NFKDemucsConvT1d(c.filters, 1, kernel: c.kernel, stride: stride,
                                                 padding: 0, bias: false)
    }

    /// Separates a mono waveform `[samples]` (`-1...1`) into `[speakers, samples]`.
    func separate(_ waveform: MLXArray) -> MLXArray {
        let samples = waveform.shape[0]
        // The released model sets `encoder_activation=None`: the filterbank output feeds the masker
        // and the mask multiplication unrectified.
        let mixture = encoder(waveform.reshaped([1, samples, 1]))         // [1, T, N]
        let time = mixture.shape[1]

        var feature = bottleneck(inputNorm(mixture))
        var skipSum = MLXArray.zeros([1, time, configuration.bottleneck])
        for block in blocks {
            let (residual, skip) = block(feature)
            feature = residual
            skipSum = skipSum + skip
        }
        // The reference's mask activation is ReLU (asteroid `TDConvNet(mask_act="relu")`), not a
        // sigmoid: masks scale the mixture rather than gating it, so they are unbounded above.
        let masks = relu(maskConv(maskPReLU(skipSum)))                    // [1, T, N·C]
            .reshaped([1, time, configuration.speakers, configuration.filters])

        var speakers: [MLXArray] = []
        for speaker in 0 ..< configuration.speakers {
            let masked = mixture * masks[0..., 0..., speaker, 0...]       // [1, T, N]
            let decoded = decoder(masked).reshaped([-1])                 // [~samples]
            speakers.append(NFKMLXConvTasNetNet.fit(decoded, to: samples).reshaped([1, samples]))
        }
        return concatenated(speakers, axis: 0)                           // [C, samples]
    }

    /// Crops or zero-pads a waveform to an exact length (the encode/decode stride can differ by a frame).
    static func fit(_ x: MLXArray, to length: Int) -> MLXArray {
        let count = x.shape[0]
        if count == length {
            return x
        }
        if count > length {
            return x[0 ..< length]
        }
        return concatenated([x, MLXArray.zeros([length - count])], axis: 0)
    }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKConvTasNetHolder: @unchecked Sendable {
    let net: NFKMLXConvTasNetNet
    init(_ net: NFKMLXConvTasNetNet) { self.net = net }
}

/// Conv-TasNet speech separation as an InferKit backend. Reads `NFKInputAudio`; returns one
/// `NFKAudioAsset` per speaker under its name ("speaker-1", "speaker-2", …).
@objc(NFKMLXConvTasNetBackend)
public final class NFKMLXConvTasNetBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKConvTasNetHolder
    private let identifier: String
    private let outputDirectory: URL

    init(net: NFKMLXConvTasNetNet, identifier: String, outputDirectory: URL = FileManager.default.temporaryDirectory) {
        holder = NFKConvTasNetHolder(net)
        self.identifier = identifier
        self.outputDirectory = outputDirectory
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, sampleRate) = Self.audio(from: request) else {
            throw NFKMLXError.unsupportedInput
        }
        let input = samples.withUnsafeBufferPointer { MLXArray($0, [samples.count]) }
        let separated = holder.net.separate(input)
        eval(separated)
        let values = separated.asArray(Float.self)                       // [C, samples] row-major
        let length = separated.shape[1]

        var outputs: [String: Any] = [:]
        for (index, name) in holder.net.configuration.speakerNames.enumerated() {
            let start = index * length
            let stream = Array(values[start ..< start + length])
            let url = outputDirectory.appendingPathComponent("tasnet-\(name)-\(UUID().uuidString).wav")
            try NFKMLXWaveFile.write(samples: stream, sampleRate: sampleRate, to: url)
            outputs[name] = NFKAudioAsset(fileURL: url, durationSeconds: Double(length) / Double(sampleRate),
                                          sampleRate: Double(sampleRate), channelCount: 1)
        }
        return NFKInferenceResult(outputs: outputs)
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        Task.detached {
            do {
                job.finish(with: try self.runInference(for: request))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    private static func audio(from request: NFKInferenceRequest) -> (samples: [Float], sampleRate: Int)? {
        guard let value = request.input(forKey: NFKInputAudio) else { return nil }
        if let asset = value as? NFKAudioAsset, let url = asset.fileURL, let data = try? Data(contentsOf: url) {
            return NFKMLXWaveFile.read(data)
        }
        if let data = value as? Data { return NFKMLXWaveFile.read(data) }
        return nil
    }
}

/// Registration and weight loading for Conv-TasNet.
@objc(NFKMLXConvTasNet)
public final class NFKMLXConvTasNet: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "conv-tasnet"

    /// Builds a Conv-TasNet speech-separation backend directly from optional local weights — no registry
    /// required. A nil `weightsURL` builds random weights (`isReady` is true).
    /// Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXConvTasNetNet(.base)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXConvTasNetBackend(net: net, identifier: modelName)
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

    /// Registers Conv-TasNet (`conv-tasnet`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a safetensors checkpoint, transposing Conv1d weights `[out, in, k]` → MLX's `[out, k, in]`.
    static func loadWeights(into net: NFKMLXConvTasNetNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        // The slope widths the module actually has, so a shared slope can be widened to match.
        let widths = Dictionary(uniqueKeysWithValues: net.parameters().flattened()
            .filter { $0.0.hasSuffix("prelu1.weight") || $0.0.hasSuffix("prelu2.weight")
                      || $0.0.hasSuffix("mask_prelu.weight") }
            .map { ($0.0, $0.1.size) })
        let mapped = raw.map { key, value -> (String, MLXArray) in
            let name = remapReferenceKey(key)
            // The global norms keep a `[1, 1, C]` shape here, while asteroid stores `[1, C, 1]`.
            if name.hasSuffix("norm1.weight") || name.hasSuffix("norm1.bias")
                || name.hasSuffix("norm2.weight") || name.hasSuffix("norm2.bias")
                || name.hasPrefix("input_norm.") {
                return (name, value.reshaped([1, 1, value.size]))
            }
            // Asteroid stores one slope per PReLU, with a trailing axis the module omits. When the
            // module is configured per-channel, that single slope is BROADCAST across the channels:
            // a constant slope applied to every channel is the same function, so a released
            // checkpoint keeps producing identical output and a fine-tune moves the slopes apart
            // from there. Widths come from the module, so this stays correct for any configuration.
            if name.hasSuffix("prelu1.weight") || name.hasSuffix("prelu2.weight")
                || name.hasSuffix("mask_prelu.weight") {
                let flat = value.reshaped([value.size])
                guard let expected = widths[name], flat.size == 1, expected > 1 else {
                    return (name, flat)
                }
                return (name, broadcast(flat, to: [expected]))
            }
            // The decoder is a transposed convolution: PyTorch stores `[in, out, kernel]`, while the
            // module's `ConvTransposed2d` over a singleton width expects `[out, kernel, 1, in]`.
            if checkpoint.needsConvTranspose, name == "decoder.conv.weight", value.ndim == 3 {
                return (name, value.transposed(1, 2, 0).expandedDimensions(axis: 2))
            }
            if checkpoint.needsConvTranspose, value.ndim == 4 { return (name, value.transposed(0, 2, 3, 1)) }
            if checkpoint.needsConvTranspose, value.ndim == 3 { return (name, value.transposed(0, 2, 1)) }
            return (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Maps an asteroid Conv-TasNet checkpoint's names onto this module's. Asteroid wraps the encoder and
    /// decoder in a `filterbank` (whose weights are `_filters`, with no bias) and packs each temporal
    /// block's six layers into a `shared_block` `nn.Sequential`, so they are addressed positionally:
    /// 0 the 1×1 convolution, 1 its PReLU, 2 the global norm, 3 the depthwise convolution, 4 its PReLU,
    /// 5 the second global norm. Its global norms name their affine terms `gamma`/`beta`.
    static func remapReferenceKey(_ key: String) -> String {
        var name = key
        name = name.replacingOccurrences(of: "encoder.filterbank._filters", with: "encoder.weight")
        name = name.replacingOccurrences(of: "decoder.filterbank._filters", with: "decoder.conv.weight")
        name = name.replacingOccurrences(of: "masker.bottleneck.0.gamma", with: "input_norm.weight")
        name = name.replacingOccurrences(of: "masker.bottleneck.0.beta", with: "input_norm.bias")
        name = name.replacingOccurrences(of: "masker.bottleneck.1.", with: "bottleneck.")
        name = name.replacingOccurrences(of: "masker.mask_net.0.weight", with: "mask_prelu.weight")
        name = name.replacingOccurrences(of: "masker.mask_net.1.", with: "mask_conv.")
        name = name.replacingOccurrences(of: "masker.TCN.", with: "blocks.")
        name = name.replacingOccurrences(of: ".res_conv.", with: ".residual.")
        name = name.replacingOccurrences(of: ".skip_conv.", with: ".skip.")
        for (index, replacement) in [("0", "conv1x1"), ("1", "prelu1"), ("2", "norm1"),
                                     ("3", "dconv"), ("4", "prelu2"), ("5", "norm2")] {
            name = name.replacingOccurrences(of: ".shared_block.\(index).", with: ".\(replacement).")
        }
        name = name.replacingOccurrences(of: "norm1.gamma", with: "norm1.weight")
        name = name.replacingOccurrences(of: "norm1.beta", with: "norm1.bias")
        name = name.replacingOccurrences(of: "norm2.gamma", with: "norm2.weight")
        name = name.replacingOccurrences(of: "norm2.beta", with: "norm2.bias")
        return name
    }
}
