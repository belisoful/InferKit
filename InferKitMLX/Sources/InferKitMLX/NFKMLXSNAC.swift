//
//  NFKMLXSNAC.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXRandom

// SNAC (Multi-Scale Neural Audio Codec): a neural audio codec whose codebooks code at DIFFERENT temporal
// rates. The residual is average-pooled before a coarse codebook quantizes it and repeat-interleaved back
// afterward, so codebook 0 emits one code per `vqStrides[0]` frames, codebook 1 per `vqStrides[1]`, and so
// on. This is the second codec here (after DAC) and the one the common speech codec-token models use.
//
// This is the 24 kHz speech model: depthwise-separable convolutions, a decoder noise block, three
// codebooks at strides [4, 2, 1], and no bottleneck attention (`attn_window_size` is null in that
// release). The Snake activation and the transposed-conv upsample are shared with the Music 3 vocoder
// (`NFKMusic3Snake`, `NFKDemucsConvT1d`); the depthwise blocks, the noise block, and the multi-scale RVQ
// are SNAC's own.
//
// The released convolutions are weight-normalized through torch's parametrization API
// (`parametrizations.weight.original0` = g, `original1` = v); the loader fuses `g·v/‖v‖`. Tensors flow as
// `[batch, length, channels]` (MLX's NLC).

/// SNAC dimensions. Defaults are the released 24 kHz speech model.
public struct NFKMLXSNACConfiguration: Sendable {
    public var sampleRate: Int
    public var encoderDim: Int
    public var encoderRates: [Int]
    public var latentDim: Int
    public var decoderDim: Int
    public var decoderRates: [Int]
    public var codebookSize: Int
    public var codebookDim: Int
    public var vqStrides: [Int]
    public var noise: Bool

    public init(sampleRate: Int = 24000, encoderDim: Int = 48, encoderRates: [Int] = [2, 4, 8, 8],
                latentDim: Int? = nil, decoderDim: Int = 1024, decoderRates: [Int] = [8, 8, 4, 2],
                codebookSize: Int = 4096, codebookDim: Int = 8, vqStrides: [Int] = [4, 2, 1],
                noise: Bool = true) {
        self.sampleRate = sampleRate
        self.encoderDim = encoderDim
        self.encoderRates = encoderRates
        self.latentDim = latentDim ?? encoderDim * (1 << encoderRates.count)
        self.decoderDim = decoderDim
        self.decoderRates = decoderRates
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
        self.vqStrides = vqStrides
        self.noise = noise
    }

    /// The released 24 kHz speech model (3 codebooks, hop 512).
    public static let snac24kHz = NFKMLXSNACConfiguration()

    /// A small configuration for weight-free tests.
    public static let tiny = NFKMLXSNACConfiguration(sampleRate: 24000, encoderDim: 8, encoderRates: [2, 4],
                                                     decoderDim: 16, decoderRates: [4, 2], codebookSize: 16,
                                                     codebookDim: 4, vqStrides: [2, 1])

    var hopLength: Int { encoderRates.reduce(1, *) }
    /// The waveform is padded to a multiple of this so every codebook's pooling divides evenly.
    var padMultiple: Int { hopLength * (vqStrides.max() ?? 1) }
}

/// A depthwise-separable residual unit: snake → depthwise dilated conv → snake → pointwise 1×1 conv,
/// added back. The depthwise convolution groups every channel, as the 24 kHz model does.
final class NFKSNACResidualUnit: Module {
    let snake1: NFKMusic3Snake
    let conv1: Conv1d
    let snake2: NFKMusic3Snake
    let conv2: Conv1d

    init(dim: Int, dilation: Int, depthwise: Bool) {
        snake1 = NFKMusic3Snake(channels: dim)
        conv1 = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7,
                       padding: (7 - 1) * dilation / 2, dilation: dilation, groups: depthwise ? dim : 1)
        snake2 = NFKMusic3Snake(channels: dim)
        conv2 = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + conv2(snake2(conv1(snake1(x))))
    }
}

/// One encoder stage: three depthwise residual units, then Snake and a strided convolution.
final class NFKSNACEncoderBlock: Module {
    @ModuleInfo(key: "res_unit1") var resUnit1: NFKSNACResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: NFKSNACResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: NFKSNACResidualUnit
    let snake: NFKMusic3Snake
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(inputDim: Int, outputDim: Int, stride: Int, depthwise: Bool) {
        _resUnit1.wrappedValue = NFKSNACResidualUnit(dim: inputDim, dilation: 1, depthwise: depthwise)
        _resUnit2.wrappedValue = NFKSNACResidualUnit(dim: inputDim, dilation: 3, depthwise: depthwise)
        _resUnit3.wrappedValue = NFKSNACResidualUnit(dim: inputDim, dilation: 9, depthwise: depthwise)
        snake = NFKMusic3Snake(channels: inputDim)
        _conv.wrappedValue = Conv1d(inputChannels: inputDim, outputChannels: outputDim, kernelSize: 2 * stride,
                                    stride: stride, padding: (stride + 1) / 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(snake(resUnit3(resUnit2(resUnit1(x)))))
    }
}

/// The SNAC encoder: a wide first convolution, the downsampling stages, and a final depthwise
/// convolution. The output width is the latent width (there is no separate projection).
final class NFKSNACEncoderNet: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [NFKSNACEncoderBlock]
    @ModuleInfo(key: "conv_out") var convOut: Conv1d

    init(_ c: NFKMLXSNACConfiguration) {
        _convIn.wrappedValue = Conv1d(inputChannels: 1, outputChannels: c.encoderDim, kernelSize: 7, padding: 3)
        var dim = c.encoderDim
        _blocks.wrappedValue = c.encoderRates.map { stride -> NFKSNACEncoderBlock in
            let input = dim
            dim *= 2
            return NFKSNACEncoderBlock(inputDim: input, outputDim: dim, stride: stride, depthwise: true)
        }
        _convOut.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7, padding: 3, groups: dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = convIn(x)
        for block in blocks { out = block(out) }
        return convOut(out)
    }
}

/// The decoder noise block: a learned per-position scale times a fresh Gaussian, added to the input.
/// Skipping it (`deterministic`) is what makes a decode reproducible, and is how parity is measured —
/// the noise's expected contribution is zero.
final class NFKSNACNoiseBlock: Module {
    @ModuleInfo(key: "linear") var linear: Conv1d

    init(dim: Int) {
        _linear.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray, deterministic: Bool) -> MLXArray {
        guard !deterministic else { return x }
        let noise = MLXRandom.normal([x.shape[0], x.shape[1], 1])
        return x + noise * linear(x)
    }
}

/// One decoder stage: snake → transposed conv → noise → three depthwise residual units.
final class NFKSNACDecoderBlock: Module {
    let snake1: NFKMusic3Snake
    @ModuleInfo(key: "conv_t1") var convT1: NFKDemucsConvT1d
    @ModuleInfo(key: "noise") var noise: NFKSNACNoiseBlock?
    @ModuleInfo(key: "res_unit1") var resUnit1: NFKSNACResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: NFKSNACResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: NFKSNACResidualUnit

    init(inputDim: Int, outputDim: Int, stride: Int, noise: Bool, depthwise: Bool) {
        snake1 = NFKMusic3Snake(channels: inputDim)
        _convT1.wrappedValue = NFKDemucsConvT1d(inputDim, outputDim, kernel: 2 * stride, stride: stride,
                                                padding: (stride + 1) / 2)
        _noise.wrappedValue = noise ? NFKSNACNoiseBlock(dim: outputDim) : nil
        _resUnit1.wrappedValue = NFKSNACResidualUnit(dim: outputDim, dilation: 1, depthwise: depthwise)
        _resUnit2.wrappedValue = NFKSNACResidualUnit(dim: outputDim, dilation: 3, depthwise: depthwise)
        _resUnit3.wrappedValue = NFKSNACResidualUnit(dim: outputDim, dilation: 9, depthwise: depthwise)
    }

    func callAsFunction(_ x: MLXArray, deterministic: Bool) -> MLXArray {
        var out = convT1(snake1(x))
        if let noise { out = noise(out, deterministic: deterministic) }
        return resUnit3(resUnit2(resUnit1(out)))
    }
}

/// The SNAC decoder: a depthwise-then-pointwise projection from the latent, the upsampling stages, then
/// Snake, a final convolution, and tanh.
final class NFKSNACDecoderNet: Module {
    @ModuleInfo(key: "conv_in_dw") var convInDW: Conv1d
    @ModuleInfo(key: "conv_in_pw") var convInPW: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [NFKSNACDecoderBlock]
    let snake: NFKMusic3Snake
    @ModuleInfo(key: "conv_out") var convOut: Conv1d

    init(_ c: NFKMLXSNACConfiguration) {
        _convInDW.wrappedValue = Conv1d(inputChannels: c.latentDim, outputChannels: c.latentDim,
                                        kernelSize: 7, padding: 3, groups: c.latentDim)
        _convInPW.wrappedValue = Conv1d(inputChannels: c.latentDim, outputChannels: c.decoderDim, kernelSize: 1)
        _blocks.wrappedValue = c.decoderRates.enumerated().map { index, stride in
            NFKSNACDecoderBlock(inputDim: c.decoderDim >> index, outputDim: c.decoderDim >> (index + 1),
                                stride: stride, noise: c.noise, depthwise: true)
        }
        let outputDim = c.decoderDim >> c.decoderRates.count
        snake = NFKMusic3Snake(channels: outputDim)
        _convOut.wrappedValue = Conv1d(inputChannels: outputDim, outputChannels: 1, kernelSize: 7, padding: 3)
    }

    func callAsFunction(_ z: MLXArray, deterministic: Bool) -> MLXArray {
        var out = convInPW(convInDW(z))
        for block in blocks { out = block(out, deterministic: deterministic) }
        return tanh(convOut(snake(out)))
    }
}

/// One multi-scale vector quantizer: pool the latent by `stride`, match each pooled frame to its nearest
/// L2-normalized codebook entry, project back, and repeat-interleave to the full rate.
final class NFKSNACVectorQuantize: Module {
    @ModuleInfo(key: "in_proj") var inProj: Conv1d
    @ModuleInfo(key: "out_proj") var outProj: Conv1d
    @ModuleInfo(key: "codebook") var codebook: Embedding

    let stride: Int

    init(inputDim: Int, codebookSize: Int, codebookDim: Int, stride: Int) {
        self.stride = stride
        _inProj.wrappedValue = Conv1d(inputChannels: inputDim, outputChannels: codebookDim, kernelSize: 1)
        _outProj.wrappedValue = Conv1d(inputChannels: codebookDim, outputChannels: inputDim, kernelSize: 1)
        _codebook.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: codebookDim)
    }

    /// `[B, T, inputDim]` → the codes `[B, T/stride]` and the full-rate reconstruction `[B, T, inputDim]`.
    func callAsFunction(_ z: MLXArray) -> (codes: MLXArray, reconstruction: MLXArray) {
        let pooled = NFKSNACVectorQuantize.avgPool(z, stride)
        let indices = NFKDACVectorQuantize.nearest(inProj(pooled), codebook: codebook.weight)   // [B, T/stride]
        let reconstruction = NFKSNACVectorQuantize.repeatInterleave(outProj(codebook(indices)), stride)
        return (indices, reconstruction)
    }

    /// Reconstructs `[B, T, inputDim]` from codes `[B, T/stride]`.
    func decode(_ codes: MLXArray) -> MLXArray {
        NFKSNACVectorQuantize.repeatInterleave(outProj(codebook(codes)), stride)
    }

    /// Non-overlapping average pool along the time axis.
    static func avgPool(_ x: MLXArray, _ stride: Int) -> MLXArray {
        guard stride > 1 else { return x }
        let (b, t, c) = (x.shape[0], x.shape[1], x.shape[2])
        return x.reshaped([b, t / stride, stride, c]).mean(axis: 2)
    }

    /// Repeat each time step `stride` times (`torch.repeat_interleave`).
    static func repeatInterleave(_ x: MLXArray, _ stride: Int) -> MLXArray {
        guard stride > 1 else { return x }
        let (b, t, c) = (x.shape[0], x.shape[1], x.shape[2])
        return broadcast(x.reshaped([b, t, 1, c]), to: [b, t, stride, c]).reshaped([b, t * stride, c])
    }
}

/// The multi-scale residual vector quantizer.
final class NFKSNACResidualVectorQuantize: Module {
    @ModuleInfo(key: "quantizers") var quantizers: [NFKSNACVectorQuantize]

    init(_ c: NFKMLXSNACConfiguration) {
        _quantizers.wrappedValue = c.vqStrides.map { stride in
            NFKSNACVectorQuantize(inputDim: c.latentDim, codebookSize: c.codebookSize,
                                  codebookDim: c.codebookDim, stride: stride)
        }
    }

    /// Latent `[B, T, latentDim]` → one code stream per codebook, each `[B, T/stride]`.
    func encode(_ z: MLXArray) -> [MLXArray] {
        var residual = z
        var codes = [MLXArray]()
        for quantizer in quantizers {
            let (indices, reconstruction) = quantizer(residual)
            residual = residual - reconstruction
            codes.append(indices)
        }
        return codes
    }

    /// Per-codebook codes → the summed quantized latent `[B, T, latentDim]`.
    func decode(_ codes: [MLXArray]) -> MLXArray {
        var z = quantizers[0].decode(codes[0])
        for index in 1 ..< quantizers.count {
            z = z + quantizers[index].decode(codes[index])
        }
        return z
    }
}

/// The SNAC codec network.
final class NFKMLXSNACNet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKSNACEncoderNet
    @ModuleInfo(key: "quantizer") var quantizer: NFKSNACResidualVectorQuantize
    @ModuleInfo(key: "decoder") var decoder: NFKSNACDecoderNet

    let configuration: NFKMLXSNACConfiguration

    init(_ c: NFKMLXSNACConfiguration) {
        configuration = c
        _encoder.wrappedValue = NFKSNACEncoderNet(c)
        _quantizer.wrappedValue = NFKSNACResidualVectorQuantize(c)
        _decoder.wrappedValue = NFKSNACDecoderNet(c)
    }

    /// A mono waveform → one code stream per codebook, at the codebook's temporal rate.
    func encode(_ samples: [Float]) -> [MLXArray] {
        let multiple = configuration.padMultiple
        var signal = samples
        let remainder = signal.count % multiple
        if remainder != 0 {
            signal += [Float](repeating: 0, count: multiple - remainder)
        }
        let audio = signal.withUnsafeBufferPointer { MLXArray($0, [1, signal.count, 1]) }
        return quantizer.encode(encoder(audio))
    }

    /// Per-codebook codes → a mono waveform `[samples]` in `-1...1`.
    func decode(_ codes: [MLXArray], deterministic: Bool) -> MLXArray {
        let wave = decoder(quantizer.decode(codes), deterministic: deterministic)
        eval(wave)
        return wave.reshaped([-1])
    }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKSNACHolder: @unchecked Sendable {
    let net: NFKMLXSNACNet
    init(_ net: NFKMLXSNACNet) { self.net = net }
}

/// SNAC reconstruction (audio → multi-scale codes → audio) as an InferKit backend, reading
/// `NFKInputAudio` and returning the reconstructed clip under `NFKOutputAudio`. The codes themselves are
/// reached through `NFKMLXSNAC.encode`/`decode`.
@objc(NFKMLXSNACBackend)
public final class NFKMLXSNACBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKSNACHolder
    private let identifier: String

    init(net: NFKMLXSNACNet, identifier: String) {
        holder = NFKSNACHolder(net)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, sampleRate) = Self.audio(from: request) else {
            throw NFKMLXError.unsupportedInput
        }
        let rate = holder.net.configuration.sampleRate
        let matched = NFKMLXAudioRate.matched(samples, from: sampleRate, to: rate)
        let waveform = holder.net.decode(holder.net.encode(matched), deterministic: false).asArray(Float.self)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("snac-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: waveform, sampleRate: rate, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(waveform.count) / Double(rate),
                                  sampleRate: Double(rate), channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
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

    private static func audio(from request: NFKInferenceRequest) -> (samples: [Float], sampleRate: Int)? {
        guard let value = request.input(forKey: NFKInputAudio) else { return nil }
        if let asset = value as? NFKAudioAsset, let url = asset.fileURL, let data = try? Data(contentsOf: url) {
            return NFKMLXWaveFile.read(data)
        }
        if let data = value as? Data { return NFKMLXWaveFile.read(data) }
        return nil
    }
}

/// Registration, encode/decode, and weight loading for SNAC.
@objc(NFKMLXSNAC)
public final class NFKMLXSNAC: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "snac"

    private let holder: NFKSNACHolder

    init(net: NFKMLXSNACNet) { holder = NFKSNACHolder(net) }

    static func makeNet(_ configuration: NFKMLXSNACConfiguration = .snac24kHz) -> NFKMLXSNACNet {
        NFKMLXSNACNet(configuration)
    }

    /// Encodes a mono waveform to its multi-scale codebook token streams. `codes[c]` is codebook `c`'s
    /// stream, at that codebook's temporal rate — the coarser codebooks emit fewer tokens.
    public func encode(_ samples: [Float]) -> [[Int]] {
        let codes = holder.net.encode(samples)
        return codes.map { stream in
            eval(stream)
            let flat = stream.reshaped([-1]).asType(.int32).asArray(Int32.self)
            return flat.map(Int.init)
        }
    }

    /// Reconstructs a mono waveform from multi-scale codebook token streams. `deterministic` skips the
    /// decoder's noise block so the output is reproducible.
    public func decode(_ codes: [[Int]], deterministic: Bool = true) -> [Float] {
        let arrays = codes.map { stream -> MLXArray in
            let flat = stream.map(Int32.init)
            return flat.withUnsafeBufferPointer { MLXArray($0, [1, stream.count]) }
        }
        return holder.net.decode(arrays, deterministic: deterministic).asArray(Float.self)
    }

    /// Builds a SNAC codec directly from optional local weights — no registry required. A nil
    /// `weightsURL` builds random weights (`isReady` is true). Run off the render thread.
    ///
    /// - Since: InferKit 0.3.1
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        NFKMLXSNACBackend(net: try loadedNet(.snac24kHz, weightsURL: weightsURL), identifier: modelName)
    }

    /// Builds a SNAC codec object (with `encode`/`decode`) directly from optional local weights.
    public static func codec(configuration: NFKMLXSNACConfiguration = .snac24kHz, weightsURL: URL?) throws -> NFKMLXSNAC {
        NFKMLXSNAC(net: try loadedNet(configuration, weightsURL: weightsURL))
    }

    private static func loadedNet(_ configuration: NFKMLXSNACConfiguration, weightsURL: URL?) throws -> NFKMLXSNACNet {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required. Blocking on the
    /// network; run off the render thread.
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

    /// Registers SNAC (`snac`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a checkpoint, fusing the parametrization-form weight norm (`g·v/‖v‖`) and transposing to
    /// MLX's layout. The reference keeps the encoder/decoder as nested `nn.Sequential` lists;
    /// `remapReferenceKey` names them.
    static func loadWeights(into net: NFKMLXSNACNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        // A fine-tuned save is already fused and in the module's layout: apply with no transpose.
        guard checkpoint.needsConvTranspose else {
            try NFKMLXWeights.apply(checkpoint.arrays.map { (remapReferenceKey($0, net.configuration), $1) }, to: net)
            return
        }
        let mapped = fusedWeightNorm(checkpoint.arrays).map { key, value -> (String, MLXArray) in
            let name = remapReferenceKey(key, net.configuration)
            if key.hasSuffix(".alpha") {
                return (name, value.transposed(0, 2, 1))
            }
            if name.contains("conv_t1"), value.ndim == 3 {
                return (name, value.transposed(1, 2, 0).expandedDimensions(axis: 2))
            }
            if value.ndim == 3 {
                return (name, value.transposed(0, 2, 1))
            }
            return (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Collapses torch parametrization pairs `X.parametrizations.weight.original0` (g) +
    /// `original1` (v) into `X.weight` (`g·v/‖v‖`, the norm over every axis but the first). SNAC's release
    /// uses the parametrization API where DAC's used the older `weight_g`/`weight_v` names.
    static func fusedWeightNorm(_ arrays: [String: MLXArray]) -> [String: MLXArray] {
        var fused = [String: MLXArray]()
        for (key, value) in arrays {
            if key.hasSuffix(".parametrizations.weight.original0") { continue }
            guard key.hasSuffix(".parametrizations.weight.original1") else {
                fused[key] = value
                continue
            }
            let base = String(key.dropLast(".parametrizations.weight.original1".count))
            guard let gain = arrays[base + ".parametrizations.weight.original0"] else {
                fused[base + ".weight"] = value
                continue
            }
            let norm = sqrt(value.square().sum(axes: Array(1 ..< value.ndim), keepDims: true))
            fused[base + ".weight"] = gain * value / maximum(norm, 1e-12)
        }
        return fused
    }

    /// Maps the reference's nested positional `nn.Sequential` names onto the module's names. The encoder
    /// is `encoder.block.N` (0 the first conv, 1…B the downsampling blocks, B+1 the final depthwise conv);
    /// each block is `block.M` (0…2 the residual units, 3 the Snake, 4 the strided conv); each residual
    /// unit is `block.K` (0 Snake, 1 the depthwise conv, 2 Snake, 3 the 1×1 conv). The decoder is
    /// `decoder.model.N` (0 the depthwise projection, 1 the pointwise projection, 2…B+1 the upsampling
    /// blocks, B+2 the Snake, B+3 the output conv); each block is `block.M` (0 Snake, 1 the transposed
    /// conv, 2 the noise block, 3…5 the residual units). The quantizer keys already match.
    static func remapReferenceKey(_ key: String, _ c: NFKMLXSNACConfiguration) -> String {
        let parts = key.split(separator: ".").map(String.init)
        if parts.first == "quantizer" { return key }

        func residualUnit(_ tail: [String]) -> String {
            guard tail.count >= 2, tail[0] == "block", let index = Int(tail[1]) else {
                return tail.joined(separator: ".")
            }
            let name = ["snake1", "conv1", "snake2", "conv2"][min(index, 3)]
            return ([name] + tail[2...]).joined(separator: ".")
        }

        if parts.first == "encoder", parts.count >= 4, parts[1] == "block", let n = Int(parts[2]) {
            let blocks = c.encoderRates.count
            if n == 0 { return "encoder.conv_in." + parts[3...].joined(separator: ".") }
            if n == blocks + 1 { return "encoder.conv_out." + parts[3...].joined(separator: ".") }
            guard parts.count >= 5, parts[3] == "block", let m = Int(parts[4]) else { return key }
            let prefix = "encoder.blocks.\(n - 1)"
            if m <= 2 { return "\(prefix).res_unit\(m + 1)." + residualUnit(Array(parts[5...])) }
            if m == 3 { return "\(prefix).snake." + parts[5...].joined(separator: ".") }
            return "\(prefix).conv." + parts[5...].joined(separator: ".")
        }

        if parts.first == "decoder", parts.count >= 3, parts[1] == "model", let n = Int(parts[2]) {
            let blocks = c.decoderRates.count
            if n == 0 { return "decoder.conv_in_dw." + parts[3...].joined(separator: ".") }
            if n == 1 { return "decoder.conv_in_pw." + parts[3...].joined(separator: ".") }
            if n == blocks + 2 { return "decoder.snake." + parts[3...].joined(separator: ".") }
            if n == blocks + 3 { return "decoder.conv_out." + parts[3...].joined(separator: ".") }
            guard parts.count >= 5, parts[3] == "block", let m = Int(parts[4]) else { return key }
            let prefix = "decoder.blocks.\(n - 2)"
            if m == 0 { return "\(prefix).snake1." + parts[5...].joined(separator: ".") }
            if m == 1 { return "\(prefix).conv_t1.conv." + parts[5...].joined(separator: ".") }
            if m == 2 { return "\(prefix).noise." + parts[5...].joined(separator: ".") }
            return "\(prefix).res_unit\(m - 2)." + residualUnit(Array(parts[5...]))
        }
        return key
    }
}
