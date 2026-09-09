//
//  NFKMLXApollo.swift
//  InferKitMLX
//
//  Apollo (JusperLee/Apollo, CC-BY-SA-4.0): music restoration of lossy-codec artifacts. An 80-band
//  split of a 20 ms STFT, each band normalized by its power and projected to a feature, six band-
//  sequence layers (a rotary transformer across the bands, then a convolutional block along time), and
//  a gated head per band back to the spectrum.
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

/// The released `configs/apollo.yaml` geometry (44.1 kHz, a 20 ms window, 256 features, 6 layers).
public struct NFKMLXApolloConfiguration: Sendable {
    public var sampleRate = 44_100
    public var windowMilliseconds = 20
    public var featureDimension = 256
    public var layers = 6
    public var heads = 8
    public var rotaryWindow = 100
    public var rotaryTheta: Float = 10_000
    public var convolutionKernel = 7
    public var window: Int { sampleRate * windowMilliseconds / 1000 }
    public var hop: Int { window / 2 }
    public var bins: Int { window / 2 + 1 }
    /// 79 bands of `window / 160` bins and a last band holding the rest.
    public var bandWidths: [Int] {
        let width = window / 160
        return [Int](repeating: width, count: 79) + [bins - 79 * width]
    }
    public init() {}
}

/// A parameter-free `nn.Sequential` entry that occupies its index.
final class NFKApolloMarker: Module {}

// MARK: - Band bottleneck and head

/// `BN[i]`: an RMS norm over the band's `2 · width + 1` channels and a 1×1 projection to the feature.
final class NFKApolloBandBottleneck: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(width: Int, feature: Int) {
        _norm.wrappedValue = RMSNorm(dimensions: 2 * width + 1, eps: 1e-5)
        _conv.wrappedValue = Conv1d(inputChannels: 2 * width + 1, outputChannels: feature, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(norm(x)) }
}

/// `output[i]`: an RMS norm, a 1×1 projection to `4 · width`, and a GLU to the band's real and
/// imaginary parts (`2 · width`, the real bins first).
final class NFKApolloBandHead: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "conv") var conv: Conv1d
    let width: Int

    init(width: Int, feature: Int) {
        self.width = width
        _norm.wrappedValue = RMSNorm(dimensions: feature, eps: 1e-5)
        _conv.wrappedValue = Conv1d(inputChannels: feature, outputChannels: 4 * width, kernelSize: 1)
    }

    /// `[B, T, feature]` → `[B, T, 2 · width]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = conv(norm(x))
        return h[0..., 0..., ..<(2 * width)] * sigmoid(h[0..., 0..., (2 * width)...])
    }
}

// MARK: - Roformer

/// The band transformer: a rotary-position self-attention (adjacent channel pairs, positions up to a
/// fixed window) over the RMS-normed input, a bias-free output projection with a residual, and a
/// gated MLP (`silu` over the whole `8 · dim` projection, then `silu(gate) · z` over its halves).
final class NFKApolloRoformer: Module {
    @ModuleInfo(key: "input_norm") var inputNorm: RMSNorm
    @ModuleInfo(key: "weight") var qkv: Conv1d
    @ModuleInfo(key: "output") var output: Conv1d
    @ModuleInfo(key: "MLP") var mlp: [Module]
    @ModuleInfo(key: "MLP_output") var mlpOutput: Conv1d
    let heads: Int
    let headDimension: Int
    let cosTable: [Float]                                                 // [window · headDimension]
    let sinTable: [Float]
    let window: Int

    init(dimension: Int, heads: Int, window: Int, theta: Float) {
        self.heads = heads
        headDimension = dimension / heads
        self.window = window
        _inputNorm.wrappedValue = RMSNorm(dimensions: dimension, eps: 1e-5)
        _qkv.wrappedValue = Conv1d(inputChannels: dimension, outputChannels: dimension * 3, kernelSize: 1, bias: false)
        _output.wrappedValue = Conv1d(inputChannels: dimension, outputChannels: dimension, kernelSize: 1, bias: false)
        _mlp.wrappedValue = [RMSNorm(dimensions: dimension, eps: 1e-5),
                             Conv1d(inputChannels: dimension, outputChannels: dimension * 8, kernelSize: 1, bias: false),
                             NFKApolloMarker()]
        _mlpOutput.wrappedValue = Conv1d(inputChannels: dimension * 4, outputChannels: dimension, kernelSize: 1, bias: false)
        // cos_freq / sin_freq: each pair of channels shares one frequency θ^(−2i/d).
        let half = headDimension / 2
        var cosValues = [Float](), sinValues = [Float]()
        for position in 0 ..< window {
            for i in 0 ..< half {
                let frequency = 1 / pow(theta, Float(2 * i) / Float(headDimension))
                let angle = Float(position) * frequency
                cosValues += [cos(angle), cos(angle)]
                sinValues += [sin(angle), sin(angle)]
            }
        }
        cosTable = cosValues
        sinTable = sinValues
    }

    /// `x [B, heads, S, d]` rotated by its position: `x · cos + (−x₂, x₁) · sin` over adjacent pairs.
    private func rotated(_ x: MLXArray) -> MLXArray {
        let s = x.dim(2)
        precondition(s <= window, "the rotary table covers \(window) positions")
        let cosine = MLXArray(Array(cosTable[0 ..< s * headDimension]), [1, 1, s, headDimension])
        let sine = MLXArray(Array(sinTable[0 ..< s * headDimension]), [1, 1, s, headDimension])
        let pairs = x.reshaped([x.dim(0), x.dim(1), s, headDimension / 2, 2])
        let negated = stacked([-pairs[0..., 0..., 0..., 0..., 1], pairs[0..., 0..., 0..., 0..., 0]], axis: -1).reshaped(x.shape)
        return x * cosine + negated * sine
    }

    /// `[B, S, dimension]` → the same shape.
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let (b, s, n) = (input.dim(0), input.dim(1), input.dim(2))
        // The projection's channels are head-major with q, k, v inside each head.
        let projected = qkv(inputNorm(input)).reshaped([b, s, heads, 3 * headDimension]).transposed(0, 2, 1, 3)
        let q = rotated(projected[0..., 0..., 0..., ..<headDimension])
        let k = rotated(projected[0..., 0..., 0..., headDimension ..< 2 * headDimension])
        let v = projected[0..., 0..., 0..., (2 * headDimension)...]
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                         scale: 1 / Float(headDimension).squareRoot(), mask: nil)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([b, s, n])
        let residual = output(merged) + input
        let norm = mlp[0] as! RMSNorm, expand = mlp[1] as! Conv1d
        let hidden = silu(expand(norm(residual)))
        let half = hidden.dim(2) / 2
        let gate = hidden[0..., 0..., ..<half], z = hidden[0..., 0..., half...]
        return residual + mlpOutput(silu(gate) * z)
    }
}

// MARK: - Sequence block

/// `ConvActNorm1d`: a depthwise 7-tap convolution, an RMS norm, a 1×1 expansion to `4 · C`, SiLU, and
/// a 1×1 projection back, added to the input (the reference `nn.Sequential`, indices 0 … 4).
final class NFKApolloConvActNorm: Module {
    @ModuleInfo(key: "conv") var conv: [Module]

    init(channels: Int, kernel: Int) {
        _conv.wrappedValue = [Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel, padding: (kernel - 1) / 2, groups: channels),
                              RMSNorm(dimensions: channels, eps: 1e-5),
                              Conv1d(inputChannels: channels, outputChannels: channels * 4, kernelSize: 1),
                              NFKApolloMarker(),
                              Conv1d(inputChannels: channels * 4, outputChannels: channels, kernelSize: 1)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let depthwise = conv[0] as! Conv1d, norm = conv[1] as! RMSNorm, expand = conv[2] as! Conv1d, project = conv[4] as! Conv1d
        return x + project(silu(expand(norm(depthwise(x)))))
    }
}

/// `ICB`: three `ConvActNorm1d` blocks along time.
final class NFKApolloICB: Module {
    @ModuleInfo(key: "blocks") var blocks: [NFKApolloConvActNorm]

    init(channels: Int, kernel: Int) {
        _blocks.wrappedValue = (0 ..< 3).map { _ in NFKApolloConvActNorm(channels: channels, kernel: kernel) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for block in blocks { h = block(h) }
        return h
    }
}

/// `BSNet`: the Roformer across the bands (every frame a sequence of 80 bands), then the ICB along
/// time (every band a sequence of frames). Runs `[B, bands, T, feature]`.
final class NFKApolloBSNet: Module {
    @ModuleInfo(key: "band_net") var bandNet: NFKApolloRoformer
    @ModuleInfo(key: "seq_net") var sequenceNet: NFKApolloICB

    init(_ c: NFKMLXApolloConfiguration) {
        _bandNet.wrappedValue = NFKApolloRoformer(dimension: c.featureDimension, heads: c.heads, window: c.rotaryWindow, theta: c.rotaryTheta)
        _sequenceNet.wrappedValue = NFKApolloICB(channels: c.featureDimension, kernel: c.convolutionKernel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, bands, t, n) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let overBands = x.transposed(0, 2, 1, 3).reshaped([b * t, bands, n])
        let communicated = bandNet(overBands).reshaped([b, t, bands, n]).transposed(0, 2, 1, 3)
        return sequenceNet(communicated.reshaped([b * bands, t, n])).reshaped([b, bands, t, n])
    }
}

// MARK: - The model

/// The Apollo network over `[channels, samples]`: the band split, the bottlenecks, the band-sequence
/// layers, the heads, and the inverse STFT at the input's length.
public final class NFKMLXApolloNet: Module {
    @ModuleInfo(key: "BN") var bottlenecks: [NFKApolloBandBottleneck]
    @ModuleInfo(key: "net") var layers: [NFKApolloBSNet]
    @ModuleInfo(key: "output") var heads: [NFKApolloBandHead]
    let configuration: NFKMLXApolloConfiguration
    let spectrum: NFKNUWaveSpectrum

    init(_ c: NFKMLXApolloConfiguration) {
        configuration = c
        spectrum = NFKNUWaveSpectrum(nFFT: c.window, hop: c.hop, normalized: false)
        _bottlenecks.wrappedValue = c.bandWidths.map { NFKApolloBandBottleneck(width: $0, feature: c.featureDimension) }
        _layers.wrappedValue = (0 ..< c.layers).map { _ in NFKApolloBSNet(c) }
        _heads.wrappedValue = c.bandWidths.map { NFKApolloBandHead(width: $0, feature: c.featureDimension) }
        super.init()
    }

    /// `feature_extractor`: each band's spectrum normalized by its power, with the log power, through
    /// its bottleneck; `[N, bands, T, feature]`.
    func features(_ signals: MLXArray) -> MLXArray {
        let (real, imaginary) = spectrum.transform(signals)                   // [N, T, bins]
        var start = 0
        var bandFeatures = [MLXArray]()
        for (i, width) in configuration.bandWidths.enumerated() {
            let re = real[0..., 0..., start ..< start + width], im = imaginary[0..., 0..., start ..< start + width]
            let power = sqrt((re.square() + im.square()).sum(axis: 2, keepDims: true) + Float.ulpOfOne)
            let concatenatedFeature = concatenated([re / power, im / power, log(power)], axis: 2)
            bandFeatures.append(bottlenecks[i](concatenatedFeature))
            start += width
        }
        return stacked(bandFeatures, axis: 1)
    }

    /// The band-sequence layers, every layer's output kept.
    func layerOutputs(_ features: MLXArray) -> [MLXArray] {
        var outputs = [MLXArray]()
        var x = features
        for layer in layers {
            x = layer(x)
            outputs.append(x)
        }
        return outputs
    }

    /// The heads' real and imaginary spectrum `[N, T, bins]` from the last layer's features.
    func estimated(_ deep: MLXArray) -> (real: MLXArray, imaginary: MLXArray) {
        var reals = [MLXArray](), imaginaries = [MLXArray]()
        for (i, width) in configuration.bandWidths.enumerated() {
            let pair = heads[i](deep[0..., i])                                  // [N, T, 2 · width]
            reals.append(pair[0..., 0..., ..<width])
            imaginaries.append(pair[0..., 0..., width...])
        }
        return (concatenated(reals, axis: 2), concatenated(imaginaries, axis: 2))
    }

    /// `[N, samples]` → `[N, samples]`.
    public func callAsFunction(_ signals: MLXArray) -> MLXArray {
        let (real, imaginary) = estimated(layerOutputs(features(signals)).last!)
        return spectrum.inverse(real: real, imaginary: imaginary, length: signals.dim(1))
    }
}

// MARK: - Backend

private final class NFKApolloHolder: @unchecked Sendable {
    let net: NFKMLXApolloNet
    init(_ net: NFKMLXApolloNet) { self.net = net }
}

/// Apollo as an InferKit backend: `NFKInputAudio` (any rate; resampled to 44.1 kHz) → the restored
/// clip under `NFKOutputAudio`. The network runs each channel on its own; the backend runs the mono
/// clip the WAV reader yields.
@objc(NFKMLXApolloBackend)
public final class NFKMLXApolloBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKApolloHolder
    private let identifier: String
    private let outputDirectory: URL

    init(net: NFKMLXApolloNet, identifier: String, outputDirectory: URL = FileManager.default.temporaryDirectory) {
        holder = NFKApolloHolder(net)
        self.identifier = identifier
        self.outputDirectory = outputDirectory
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, rate) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
        let configuration = holder.net.configuration
        let matched = rate == configuration.sampleRate ? samples : NFKMLXAudioRate.matched(samples, from: rate, to: configuration.sampleRate)
        let restored = Self.restore(matched, net: holder.net)
        let url = outputDirectory.appendingPathComponent("apollo-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: restored, sampleRate: configuration.sampleRate, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(restored.count) / Double(configuration.sampleRate),
                                  sampleRate: Double(configuration.sampleRate), channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
    }

    static func restore(_ samples: [Float], net: NFKMLXApolloNet) -> [Float] {
        let signal = samples.withUnsafeBufferPointer { MLXArray($0, [1, samples.count]) }
        let output = net(signal)
        eval(output)
        return output.reshaped([-1]).asArray(Float.self)
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        Task.detached(priority: .userInitiated) {
            do { job.finish(with: try self.runInference(for: request)) }
            catch { job.finish(withError: error as NSError) }
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

/// Registration and weight loading for Apollo.
@objc(NFKMLXApollo)
public final class NFKMLXApollo: NSObject {
    @objc public static let modelName = "apollo"

    static func makeNet(_ configuration: NFKMLXApolloConfiguration = .init()) -> NFKMLXApolloNet {
        NFKMLXApolloNet(configuration)
    }

    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL { try loadWeights(into: net, from: weightsURL) }
        return NFKMLXApolloBackend(net: net, identifier: modelName)
    }

    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) }, completionHandler: completionHandler)
    }

    /// Registers `apollo` with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads the released `pytorch_model.bin` (its `state_dict` through the native torch reader). The
    /// rotary tables are recomputed; the band bottlenecks' and heads' `nn.Sequential` indices become
    /// `norm` / `conv`; the 1-D convolutions transpose to channels-last.
    static func loadWeights(into net: NFKMLXApolloNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped: [(String, MLXArray)] = checkpoint.arrays.compactMap { key, value in
            guard let name = remapReferenceKey(key) else { return nil }
            let tensor = value.ndim == 3 && checkpoint.needsConvTranspose ? value.transposed(0, 2, 1) : value
            return (name, tensor)
        }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }

    static func remapReferenceKey(_ key: String) -> String? {
        if key.hasSuffix("cos_freq") || key.hasSuffix("sin_freq") { return nil }
        var name = key
        for stem in ["BN.", "output."] where name.hasPrefix(stem) {
            name = name.replacingOccurrences(of: #"^(BN|output)\.(\d+)\.0\."#, with: "$1.$2.norm.", options: .regularExpression)
            name = name.replacingOccurrences(of: #"^(BN|output)\.(\d+)\.1\."#, with: "$1.$2.conv.", options: .regularExpression)
        }
        return name
    }
}
