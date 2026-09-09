//
//  NFKMLXFRCRN.swift
//  InferKitMLX
//
//  FRCRN (modelscope/ClearerVoice-Studio, Apache-2.0): a frequency-recurrent complex convolutional
//  recurrent network. Two complex UNets over a convolutional STFT estimate a complex ratio mask; each
//  UNet threads a frequency-recurrent FSMN memory between its stages and a complex squeeze-excite
//  after each encoder.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// The released FRCRN_SE_16K geometry (`model_depth` 14 at 16 kHz).
public struct NFKMLXFRCRNConfiguration: Sendable {
    public var sampleRate = 16_000
    public var fftSize = 640
    public var hopSize = 320
    public var channels = 128
    /// The FSMN memory order (`lorder`).
    public var memoryOrder = 20
    /// Encoder kernels over `(frequency, time)`; every stride is `(2, 1)` at padding `(0, 1)`.
    public var encoderKernels: [(Int, Int)] = [(5, 2), (5, 2), (5, 2), (5, 2), (5, 2), (5, 2), (2, 2)]
    public var decoderKernels: [(Int, Int)] = [(2, 2), (5, 2), (5, 2), (5, 2), (6, 2), (5, 2), (5, 2)]
    /// The squeeze-excite bottleneck ratio.
    public var squeezeReduction = 8
    /// `decode_one_audio_frcrn_se_16k`'s padding grid: a one-second window and a 0.75 s stride.
    public var decodeWindow = 16_000
    public var decodeStride = 12_000
    public var bins: Int { fftSize / 2 + 1 }
    public var stages: Int { encoderKernels.count }
    public init() {}
}

/// A complex activation as two real arrays, both `[B, frequency, time, channels]`.
struct NFKFRCRNComplex {
    var real: MLXArray
    var imaginary: MLXArray
}

/// A parameter-free `nn.Sequential` entry that occupies its index.
final class NFKFRCRNMarker: Module {}

// MARK: - Complex layers

/// `ComplexConv2d`: `re = conv_re(x_re) − conv_im(x_im)`, `im = conv_re(x_im) + conv_im(x_re)`.
final class NFKFRCRNComplexConv: Module {
    @ModuleInfo(key: "conv_re") var re: Conv2d
    @ModuleInfo(key: "conv_im") var im: Conv2d

    init(_ input: Int, _ output: Int, kernel: (Int, Int), stride: (Int, Int) = (1, 1), padding: (Int, Int) = (0, 0)) {
        _re.wrappedValue = Conv2d(inputChannels: input, outputChannels: output, kernelSize: IntOrPair(kernel),
                                  stride: IntOrPair(stride), padding: IntOrPair(padding))
        _im.wrappedValue = Conv2d(inputChannels: input, outputChannels: output, kernelSize: IntOrPair(kernel),
                                  stride: IntOrPair(stride), padding: IntOrPair(padding))
    }

    func callAsFunction(_ x: NFKFRCRNComplex) -> NFKFRCRNComplex {
        NFKFRCRNComplex(real: re(x.real) - im(x.imaginary), imaginary: re(x.imaginary) + im(x.real))
    }
}

/// `ComplexConvTranspose2d`, the same rule over transposed convolutions.
final class NFKFRCRNComplexConvTransposed: Module {
    @ModuleInfo(key: "tconv_re") var re: ConvTransposed2d
    @ModuleInfo(key: "tconv_im") var im: ConvTransposed2d

    init(_ input: Int, _ output: Int, kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int)) {
        _re.wrappedValue = ConvTransposed2d(inputChannels: input, outputChannels: output, kernelSize: IntOrPair(kernel),
                                            stride: IntOrPair(stride), padding: IntOrPair(padding))
        _im.wrappedValue = ConvTransposed2d(inputChannels: input, outputChannels: output, kernelSize: IntOrPair(kernel),
                                            stride: IntOrPair(stride), padding: IntOrPair(padding))
    }

    func callAsFunction(_ x: NFKFRCRNComplex) -> NFKFRCRNComplex {
        NFKFRCRNComplex(real: re(x.real) - im(x.imaginary), imaginary: re(x.imaginary) + im(x.real))
    }
}

/// `ComplexBatchNorm2d`: independent BatchNorms over the real and imaginary parts.
final class NFKFRCRNComplexBatchNorm: Module {
    @ModuleInfo(key: "bn_re") var re: BatchNorm
    @ModuleInfo(key: "bn_im") var im: BatchNorm

    init(_ channels: Int) {
        _re.wrappedValue = BatchNorm(featureCount: channels)
        _im.wrappedValue = BatchNorm(featureCount: channels)
    }

    func callAsFunction(_ x: NFKFRCRNComplex) -> NFKFRCRNComplex {
        NFKFRCRNComplex(real: re(x.real), imaginary: im(x.imaginary))
    }
}

/// `Encoder`: a complex convolution, a complex BatchNorm, LeakyReLU (0.01) on both parts.
final class NFKFRCRNEncoder: Module {
    @ModuleInfo(key: "conv") var conv: NFKFRCRNComplexConv
    @ModuleInfo(key: "bn") var norm: NFKFRCRNComplexBatchNorm

    init(_ input: Int, _ output: Int, kernel: (Int, Int)) {
        _conv.wrappedValue = NFKFRCRNComplexConv(input, output, kernel: kernel, stride: (2, 1), padding: (0, 1))
        _norm.wrappedValue = NFKFRCRNComplexBatchNorm(output)
    }

    func callAsFunction(_ x: NFKFRCRNComplex) -> NFKFRCRNComplex {
        let h = norm(conv(x))
        return NFKFRCRNComplex(real: leakyRelu(h.real, negativeSlope: 0.01), imaginary: leakyRelu(h.imaginary, negativeSlope: 0.01))
    }
}

/// `Decoder`: a complex transposed convolution, a complex BatchNorm, LeakyReLU (0.01).
final class NFKFRCRNDecoder: Module {
    @ModuleInfo(key: "transconv") var conv: NFKFRCRNComplexConvTransposed
    @ModuleInfo(key: "bn") var norm: NFKFRCRNComplexBatchNorm

    init(_ input: Int, _ output: Int, kernel: (Int, Int)) {
        _conv.wrappedValue = NFKFRCRNComplexConvTransposed(input, output, kernel: kernel, stride: (2, 1), padding: (0, 1))
        _norm.wrappedValue = NFKFRCRNComplexBatchNorm(output)
    }

    func callAsFunction(_ x: NFKFRCRNComplex) -> NFKFRCRNComplex {
        let h = norm(conv(x))
        return NFKFRCRNComplex(real: leakyRelu(h.real, negativeSlope: 0.01), imaginary: leakyRelu(h.imaginary, negativeSlope: 0.01))
    }
}

// MARK: - FSMN memory

/// `UniDeepFsmn`: `linear` → ReLU → a bias-free `project`, then a causal depthwise memory of
/// `memoryOrder` taps along the sequence axis (`conv1`, left-padded so a step reads only its past),
/// added back to the projection, the whole added back to the input. Runs `[N, sequence, features]`.
final class NFKFRCRNFSMN: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "project") var project: Linear
    @ModuleInfo(key: "conv1") var memory: Conv1d
    let order: Int

    init(_ input: Int, hidden: Int, output: Int, order: Int) {
        self.order = order
        _linear.wrappedValue = Linear(input, hidden)
        _project.wrappedValue = Linear(hidden, output, bias: false)
        _memory.wrappedValue = Conv1d(inputChannels: output, outputChannels: output, kernelSize: order, groups: output, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = project(relu(linear(x)))
        let padded = MLX.padded(projected, widths: [IntOrPair(0), IntOrPair((order - 1, 0)), IntOrPair(0)], mode: .constant)
        return x + projected + memory(padded)
    }
}

/// One complex FSMN layer: `re = f_re(x_re) − f_im(x_im)`, `im = f_re(x_im) + f_im(x_re)`.
struct NFKFRCRNComplexFSMNLayer {
    let re: NFKFRCRNFSMN
    let im: NFKFRCRNFSMN

    func callAsFunction(_ real: MLXArray, _ imaginary: MLXArray) -> (MLXArray, MLXArray) {
        (re(real) - im(imaginary), re(imaginary) + im(real))
    }
}

/// `ComplexUniDeepFsmn_L1`: one complex FSMN layer whose SEQUENCE is the frequency axis, each
/// frame its own sequence (the frequency recurrence FRCRN is named for).
final class NFKFRCRNFrequencyFSMN: Module {
    @ModuleInfo(key: "fsmn_re_L1") var re: NFKFRCRNFSMN
    @ModuleInfo(key: "fsmn_im_L1") var im: NFKFRCRNFSMN

    init(_ channels: Int, order: Int) {
        _re.wrappedValue = NFKFRCRNFSMN(channels, hidden: channels, output: channels, order: order)
        _im.wrappedValue = NFKFRCRNFSMN(channels, hidden: channels, output: channels, order: order)
    }

    func callAsFunction(_ x: NFKFRCRNComplex) -> NFKFRCRNComplex {
        let (b, d, t, c) = (x.real.dim(0), x.real.dim(1), x.real.dim(2), x.real.dim(3))
        func sequences(_ a: MLXArray) -> MLXArray { a.transposed(0, 2, 1, 3).reshaped([b * t, d, c]) }
        func restore(_ a: MLXArray) -> MLXArray { a.reshaped([b, t, d, c]).transposed(0, 2, 1, 3) }
        let (real, imaginary) = NFKFRCRNComplexFSMNLayer(re: re, im: im)(sequences(x.real), sequences(x.imaginary))
        return NFKFRCRNComplex(real: restore(real), imaginary: restore(imaginary))
    }
}

/// `ComplexUniDeepFsmn`: two complex FSMN layers over TIME at the bottleneck, where the frequency
/// axis has collapsed to one bin and the features are the channels.
final class NFKFRCRNTimeFSMN: Module {
    @ModuleInfo(key: "fsmn_re_L1") var re1: NFKFRCRNFSMN
    @ModuleInfo(key: "fsmn_im_L1") var im1: NFKFRCRNFSMN
    @ModuleInfo(key: "fsmn_re_L2") var re2: NFKFRCRNFSMN
    @ModuleInfo(key: "fsmn_im_L2") var im2: NFKFRCRNFSMN

    init(_ channels: Int, order: Int) {
        _re1.wrappedValue = NFKFRCRNFSMN(channels, hidden: channels, output: channels, order: order)
        _im1.wrappedValue = NFKFRCRNFSMN(channels, hidden: channels, output: channels, order: order)
        _re2.wrappedValue = NFKFRCRNFSMN(channels, hidden: channels, output: channels, order: order)
        _im2.wrappedValue = NFKFRCRNFSMN(channels, hidden: channels, output: channels, order: order)
    }

    func callAsFunction(_ x: NFKFRCRNComplex) -> NFKFRCRNComplex {
        let (b, d, t, c) = (x.real.dim(0), x.real.dim(1), x.real.dim(2), x.real.dim(3))
        precondition(d == 1, "the bottleneck FSMN reads one frequency bin; the feature order (channel outer, bin inner) only collapses at d == 1")
        func sequences(_ a: MLXArray) -> MLXArray { a.transposed(0, 2, 1, 3).reshaped([b, t, d * c]) }
        func restore(_ a: MLXArray) -> MLXArray { a.reshaped([b, t, d, c]).transposed(0, 2, 1, 3) }
        let (r1, i1) = NFKFRCRNComplexFSMNLayer(re: re1, im: im1)(sequences(x.real), sequences(x.imaginary))
        let (r2, i2) = NFKFRCRNComplexFSMNLayer(re: re2, im: im2)(r1, i1)
        return NFKFRCRNComplex(real: restore(r2), imaginary: restore(i2))
    }
}

// MARK: - Squeeze-excite

/// `SELayer`: the real and imaginary parts are each average-pooled over frequency and time, run
/// through their own two-layer MLPs (`fc_r`, `fc_i`: Linear, ReLU, Linear, Sigmoid), combined as a
/// complex product of the two gates, and then scale the input PART BY PART (`x_re · y_re`,
/// `x_im · y_im`, an elementwise scale rather than a complex multiply).
final class NFKFRCRNSqueezeExcite: Module {
    @ModuleInfo(key: "fc_r") var real: [Module]
    @ModuleInfo(key: "fc_i") var imaginary: [Module]

    init(_ channels: Int, reduction: Int) {
        _real.wrappedValue = [Linear(channels, channels / reduction), NFKFRCRNMarker(), Linear(channels / reduction, channels), NFKFRCRNMarker()]
        _imaginary.wrappedValue = [Linear(channels, channels / reduction), NFKFRCRNMarker(), Linear(channels / reduction, channels), NFKFRCRNMarker()]
    }

    private static func gate(_ stack: [Module], _ x: MLXArray) -> MLXArray {
        let first = stack[0] as! Linear, second = stack[2] as! Linear
        return sigmoid(second(relu(first(x))))
    }

    func callAsFunction(_ x: NFKFRCRNComplex) -> NFKFRCRNComplex {
        let pooledRe = x.real.mean(axes: [1, 2])                                 // [B, C]
        let pooledIm = x.imaginary.mean(axes: [1, 2])
        let gateRe = Self.gate(real, pooledRe) - Self.gate(imaginary, pooledIm)
        let gateIm = Self.gate(real, pooledIm) + Self.gate(imaginary, pooledRe)
        let c = x.real.dim(3)
        return NFKFRCRNComplex(real: x.real * gateRe.reshaped([-1, 1, 1, c]),
                               imaginary: x.imaginary * gateIm.reshaped([-1, 1, 1, c]))
    }
}

// MARK: - UNet

/// The FRCRN complex UNet: seven encoders (a frequency FSMN before each but the first, a squeeze-excite
/// after each), the two-layer time FSMN at the bottleneck, seven decoders (a frequency FSMN after each
/// but the last, a squeeze-excite after the first five, the matching encoder's excited output
/// concatenated on channels), and a complex 1×1 `linear`.
final class NFKFRCRNUNet: Module {
    @ModuleInfo(key: "encoders") var encoders: [NFKFRCRNEncoder]
    @ModuleInfo(key: "fsmn_enc") var encoderMemories: [NFKFRCRNFrequencyFSMN]
    @ModuleInfo(key: "se_layers_enc") var encoderExcites: [NFKFRCRNSqueezeExcite]
    @ModuleInfo(key: "fsmn") var bottleneck: NFKFRCRNTimeFSMN
    @ModuleInfo(key: "decoders") var decoders: [NFKFRCRNDecoder]
    @ModuleInfo(key: "fsmn_dec") var decoderMemories: [NFKFRCRNFrequencyFSMN]
    @ModuleInfo(key: "se_layers_dec") var decoderExcites: [NFKFRCRNSqueezeExcite]
    @ModuleInfo(key: "linear") var linear: NFKFRCRNComplexConv
    let stages: Int

    init(_ c: NFKMLXFRCRNConfiguration) {
        let stages = c.stages
        self.stages = stages
        let width = c.channels
        _encoders.wrappedValue = (0 ..< stages).map { NFKFRCRNEncoder($0 == 0 ? 1 : width, width, kernel: c.encoderKernels[$0]) }
        _encoderMemories.wrappedValue = (0 ..< stages).map { _ in NFKFRCRNFrequencyFSMN(width, order: c.memoryOrder) }
        _encoderExcites.wrappedValue = (0 ..< stages).map { _ in NFKFRCRNSqueezeExcite(width, reduction: c.squeezeReduction) }
        _bottleneck.wrappedValue = NFKFRCRNTimeFSMN(width, order: c.memoryOrder)
        // The reference's dec_channels: 64 → 128 … 128 → 1, each decoder reading twice its stage width.
        _decoders.wrappedValue = (0 ..< stages).map {
            NFKFRCRNDecoder($0 == 0 ? width : width * 2, $0 == stages - 1 ? 1 : width, kernel: c.decoderKernels[$0])
        }
        _decoderMemories.wrappedValue = (0 ..< stages).map { _ in NFKFRCRNFrequencyFSMN(width, order: c.memoryOrder) }
        _decoderExcites.wrappedValue = (0 ..< stages - 1).map { _ in NFKFRCRNSqueezeExcite(width, reduction: c.squeezeReduction) }
        _linear.wrappedValue = NFKFRCRNComplexConv(1, 1, kernel: (1, 1))
        super.init()
    }

    func callAsFunction(_ input: NFKFRCRNComplex) -> NFKFRCRNComplex {
        stagesOutput(input).output
    }

    /// The forward with the seams the parity harness reads.
    func stagesOutput(_ input: NFKFRCRNComplex) -> (encoder0: NFKFRCRNComplex, excited0: NFKFRCRNComplex,
                                                    bottleneck: NFKFRCRNComplex, decoder0: NFKFRCRNComplex, output: NFKFRCRNComplex) {
        var x = input
        var excited = [input]
        var encoder0 = input
        for i in 0 ..< stages {
            if i > 0 { x = encoderMemories[i](x) }
            x = encoders[i](x)
            if i == 0 { encoder0 = x }
            excited.append(encoderExcites[i](x))
        }
        x = bottleneck(x)
        let bottleneckOut = x
        var p = x
        var decoder0 = x
        for i in 0 ..< stages {
            p = decoders[i](p)
            if i == 0 { decoder0 = p }
            if i < stages - 1 { p = decoderMemories[i](p) }
            if i == stages - 1 { break }
            if i < stages - 2 { p = decoderExcites[i](p) }
            let skip = excited[stages - 1 - i]
            p = NFKFRCRNComplex(real: concatenated([p.real, skip.real], axis: 3),
                                imaginary: concatenated([p.imaginary, skip.imaginary], axis: 3))
        }
        return (encoder0, excited[1], bottleneckOut, decoder0, linear(p))
    }
}

// MARK: - The model

/// FRCRN's `DCCRN`: two UNets, the second reading the first's raw output; the mask is
/// `tanh(unet1) + tanh(unet2)`, applied as a complex product to the spectrum.
public final class NFKMLXFRCRNNet: Module {
    @ModuleInfo(key: "unet") var first: NFKFRCRNUNet
    @ModuleInfo(key: "unet2") var second: NFKFRCRNUNet
    let configuration: NFKMLXFRCRNConfiguration

    init(_ c: NFKMLXFRCRNConfiguration) {
        configuration = c
        _first.wrappedValue = NFKFRCRNUNet(c)
        _second.wrappedValue = NFKFRCRNUNet(c)
        super.init()
    }

    /// The first UNet's output and the combined mask, both `[B, bins, frames, 1]` pairs.
    func mask(_ spectrum: NFKFRCRNComplex) -> (unet1: NFKFRCRNComplex, mask: NFKFRCRNComplex) {
        let unet1 = first(spectrum)
        let unet2 = second(unet1)
        return (unet1, NFKFRCRNComplex(real: tanh(unet2.real) + tanh(unet1.real),
                                       imaginary: tanh(unet2.imaginary) + tanh(unet1.imaginary)))
    }

    /// The masked spectrum: the spectrum times the mask as complex numbers.
    func masked(_ spectrum: NFKFRCRNComplex) -> NFKFRCRNComplex {
        let m = mask(spectrum).mask
        return NFKFRCRNComplex(real: spectrum.real * m.real - spectrum.imaginary * m.imaginary,
                               imaginary: spectrum.real * m.imaginary + spectrum.imaginary * m.real)
    }
}

// MARK: - Backend

private final class NFKFRCRNHolder: @unchecked Sendable {
    let net: NFKMLXFRCRNNet
    init(_ net: NFKMLXFRCRNNet) { self.net = net }
}

/// FRCRN as an InferKit backend: `NFKInputAudio` (any rate; resampled to 16 kHz) → the enhanced clip
/// under `NFKOutputAudio`.
@objc(NFKMLXFRCRNBackend)
public final class NFKMLXFRCRNBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKFRCRNHolder
    private let identifier: String
    private let outputDirectory: URL

    init(net: NFKMLXFRCRNNet, identifier: String, outputDirectory: URL = FileManager.default.temporaryDirectory) {
        holder = NFKFRCRNHolder(net)
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
        let matched = NFKMLXAudioRate.matched(samples, from: rate, to: configuration.sampleRate)
        let enhanced = Self.enhance(matched, net: holder.net)
        eval(enhanced)
        let stream = enhanced.reshaped([enhanced.shape.last!]).asArray(Float.self)
        let url = outputDirectory.appendingPathComponent("frcrn-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: stream, sampleRate: configuration.sampleRate, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(stream.count) / Double(configuration.sampleRate),
                                  sampleRate: Double(configuration.sampleRate), channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
    }

    /// The reference `ConvSTFT`: a square-root periodic Hann of the FFT size, no centering.
    static func stft(_ configuration: NFKMLXFRCRNConfiguration) -> NFKMLXComplexSTFT {
        let window = MLXArray(nfkPeriodicHann(configuration.fftSize).map { $0.squareRoot() })
        return NFKMLXComplexSTFT(nFFT: configuration.fftSize, hop: configuration.hopSize, window: window, centered: false)
    }

    /// `decode_one_audio_frcrn_se_16k`'s zero padding: up to the window, up to window + stride, or
    /// (past that) by `t − ⌊(t − window) / stride⌋ · stride` whenever the clip is off the stride grid.
    /// The network's frequency memories and global pools read the padded clip, so the padding is part
    /// of the model's input rather than a convenience.
    static func padded(_ samples: [Float], configuration: NFKMLXFRCRNConfiguration) -> [Float] {
        let t = samples.count, window = configuration.decodeWindow, stride = configuration.decodeStride
        let pad: Int
        if t < window {
            pad = window - t
        } else if t < window + stride {
            pad = window + stride - t
        } else {
            pad = (t - window) % stride != 0 ? t - (t - window) / stride * stride : 0
        }
        return samples + [Float](repeating: 0, count: pad)
    }

    /// The conv-STFT spectrum of a prepared clip, real and imaginary each `[1, bins, frames, 1]`.
    static func spectrum(_ padded: [Float], configuration: NFKMLXFRCRNConfiguration) -> NFKFRCRNComplex {
        let signal = padded.withUnsafeBufferPointer { MLXArray($0, [1, padded.count]) }
        let (real, imaginary) = stft(configuration).transformComplex(signal)      // [1, bins, frames]
        return NFKFRCRNComplex(real: real.expandedDimensions(axis: 3), imaginary: imaginary.expandedDimensions(axis: 3))
    }

    /// `inference` over the padded clip, trimmed back to the input length.
    static func enhance(_ samples: [Float], net: NFKMLXFRCRNNet) -> MLXArray {
        let configuration = net.configuration
        let masked = net.masked(spectrum(padded(samples, configuration: configuration), configuration: configuration))
        let waveform = stft(configuration).inverseComplex(real: masked.real.squeezed(axis: 3), imaginary: masked.imaginary.squeezed(axis: 3))
        return waveform[0..., 0 ..< min(samples.count, waveform.dim(1))]
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

/// Registration and weight loading for FRCRN.
@objc(NFKMLXFRCRN)
public final class NFKMLXFRCRN: NSObject {
    @objc public static let modelName = "frcrn"

    static func makeNet(_ configuration: NFKMLXFRCRNConfiguration = .init()) -> NFKMLXFRCRNNet {
        let net = NFKMLXFRCRNNet(configuration)
        net.train(false)                                                        // BatchNorm reads its running statistics
        return net
    }

    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL { try loadWeights(into: net, from: weightsURL) }
        return NFKMLXFRCRNBackend(net: net, identifier: modelName)
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

    /// Registers `frcrn` with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads the released `last_best_checkpoint.pt` (its `model` dict through the native torch reader).
    /// The checkpoint carries every stage TWICE, as the flat `encoder{i}` / `decoder{i}` /
    /// `se_layer_enc{i}` / `fsmn_enc{i}` attributes and again as the `ModuleList`s `encoders.{i}` …;
    /// the lists are what the module is keyed by and the flat copies are dropped. The conv-STFT
    /// kernels are dropped too (the transform is computed). Convolution weights transpose to
    /// channels-last, the transposed convolutions through `(1, 2, 3, 0)`, and the FSMN's `[C, 1, order, 1]`
    /// depthwise memory to the 1-D `[C, order, 1]`.
    static func loadWeights(into net: NFKMLXFRCRNNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped: [(String, MLXArray)] = checkpoint.arrays.compactMap { key, value in
            guard let name = remapReferenceKey(key) else { return nil }
            var tensor = value
            if checkpoint.needsConvTranspose {
                if name.hasSuffix("conv1.weight") {
                    tensor = value.reshaped([value.dim(0), value.dim(1), value.dim(2)]).transposed(0, 2, 1)
                } else if value.ndim == 4 {
                    tensor = name.contains("tconv_") ? value.transposed(1, 2, 3, 0) : value.transposed(0, 2, 3, 1)
                }
            }
            return (name, tensor)
        }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }

    static func remapReferenceKey(_ key: String) -> String? {
        if key.hasSuffix("num_batches_tracked") || key.hasPrefix("stft.") || key.hasPrefix("istft.") { return nil }
        var name = key
        if name.hasPrefix("model.") { name = String(name.dropFirst(6)) }
        // The flat attribute copies (`unet.encoder3.`, `unet.fsmn_dec0.`) duplicate the lists.
        for stem in ["encoder", "decoder", "se_layer_enc", "se_layer_dec", "fsmn_enc", "fsmn_dec"] {
            for prefix in ["unet.", "unet2."] where name.hasPrefix(prefix + stem) {
                let rest = name.dropFirst(prefix.count + stem.count)
                if let first = rest.first, first.isNumber { return nil }
            }
        }
        return name
    }
}
