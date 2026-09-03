//
//  NFKMLXDAC.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// The Descript Audio Codec (DAC): a neural audio codec that compresses a waveform into a small stack of
// integer token streams and reconstructs it. It is the first codec in the toolkit, and the class a
// codec-token speech-LLM generates into. Three parts: a convolutional Encoder that downsamples the
// waveform to a latent sequence, a Residual Vector Quantizer (RVQ) that turns each latent frame into
// `nCodebooks` integer codes, and a Decoder that reconstructs the waveform from the quantized latent.
//
// The Snake activation, the dilated residual unit, and the decoder's upsample block are shared with the
// Music 3 vocoder, which is itself a DAC-style Snake decoder (`NFKMusic3Snake`, `NFKMusic3ResidualUnit`,
// `NFKMusic3VocoderBlock`). The encoder mirror and the RVQ are the parts specific to the codec.
//
// The released convolutions are weight-normalized; the loader fuses `g·v/‖v‖` the way the Music 3
// vocoder loader does. Tensors flow as `[batch, length, channels]` (MLX's NLC).

/// DAC dimensions. Defaults are the released 44.1 kHz model.
public struct NFKMLXDACConfiguration: Sendable {
    public var sampleRate: Int
    public var encoderDim: Int
    public var encoderRates: [Int]
    public var latentDim: Int
    public var decoderDim: Int
    public var decoderRates: [Int]
    public var codebooks: Int
    public var codebookSize: Int
    public var codebookDim: Int

    public init(sampleRate: Int = 44100, encoderDim: Int = 64, encoderRates: [Int] = [2, 4, 8, 8],
                latentDim: Int? = nil, decoderDim: Int = 1536, decoderRates: [Int] = [8, 8, 4, 2],
                codebooks: Int = 9, codebookSize: Int = 1024, codebookDim: Int = 8) {
        self.sampleRate = sampleRate
        self.encoderDim = encoderDim
        self.encoderRates = encoderRates
        // The reference derives the latent width from the encoder geometry when it is not given.
        self.latentDim = latentDim ?? encoderDim * (1 << encoderRates.count)
        self.decoderDim = decoderDim
        self.decoderRates = decoderRates
        self.codebooks = codebooks
        self.codebookSize = codebookSize
        self.codebookDim = codebookDim
    }

    /// The released 44.1 kHz model (9 codebooks, hop 512).
    public static let dac44kHz = NFKMLXDACConfiguration()

    /// The released 24 kHz model (32 codebooks, hop 320).
    public static let dac24kHz = NFKMLXDACConfiguration(sampleRate: 24000, encoderRates: [2, 4, 5, 8],
                                                        decoderRates: [8, 5, 4, 2], codebooks: 32)

    /// The released 16 kHz model (12 codebooks, hop 320).
    public static let dac16kHz = NFKMLXDACConfiguration(sampleRate: 16000, encoderRates: [2, 4, 5, 8],
                                                        decoderRates: [8, 5, 4, 2], codebooks: 12)

    /// A small configuration for weight-free tests. Hop 8, so a forward is cheap.
    public static let tiny = NFKMLXDACConfiguration(sampleRate: 16000, encoderDim: 8, encoderRates: [2, 4],
                                                    decoderDim: 16, decoderRates: [4, 2], codebooks: 2,
                                                    codebookSize: 16, codebookDim: 4)

    /// The factor the encoder downsamples the waveform by.
    var hopLength: Int { encoderRates.reduce(1, *) }
}

/// One encoder stage: three dilated residual units at the input width, then Snake and a strided
/// convolution that halves the resolution and doubles the width.
final class NFKDACEncoderBlock: Module {
    @ModuleInfo(key: "res_unit1") var resUnit1: NFKMusic3ResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: NFKMusic3ResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: NFKMusic3ResidualUnit
    let snake: NFKMusic3Snake
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(dim: Int, stride: Int) {
        _resUnit1.wrappedValue = NFKMusic3ResidualUnit(dim: dim / 2, dilation: 1)
        _resUnit2.wrappedValue = NFKMusic3ResidualUnit(dim: dim / 2, dilation: 3)
        _resUnit3.wrappedValue = NFKMusic3ResidualUnit(dim: dim / 2, dilation: 9)
        snake = NFKMusic3Snake(channels: dim / 2)
        _conv.wrappedValue = Conv1d(inputChannels: dim / 2, outputChannels: dim, kernelSize: 2 * stride,
                                    stride: stride, padding: (stride + 1) / 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(snake(resUnit3(resUnit2(resUnit1(x)))))
    }
}

/// The DAC encoder: a wide first convolution, the downsampling stages, then Snake and a projection to
/// the latent width.
final class NFKDACEncoderNet: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [NFKDACEncoderBlock]
    let snake: NFKMusic3Snake
    @ModuleInfo(key: "conv_out") var convOut: Conv1d

    init(_ c: NFKMLXDACConfiguration) {
        _convIn.wrappedValue = Conv1d(inputChannels: 1, outputChannels: c.encoderDim, kernelSize: 7, padding: 3)
        var dim = c.encoderDim
        _blocks.wrappedValue = c.encoderRates.map { stride -> NFKDACEncoderBlock in
            dim *= 2
            return NFKDACEncoderBlock(dim: dim, stride: stride)
        }
        snake = NFKMusic3Snake(channels: dim)
        _convOut.wrappedValue = Conv1d(inputChannels: dim, outputChannels: c.latentDim, kernelSize: 3, padding: 1)
    }

    /// `[B, samples, 1]` → latent `[B, frames, latentDim]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = convIn(x)
        for block in blocks {
            out = block(out)
        }
        return convOut(snake(out))
    }
}

/// The DAC decoder: a projection from the latent width, the upsampling stages, then Snake, a final
/// convolution, and tanh. The upsample block is the shared Music 3 vocoder block.
final class NFKDACDecoderNet: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [NFKMusic3VocoderBlock]
    let snake: NFKMusic3Snake
    @ModuleInfo(key: "conv_out") var convOut: Conv1d

    init(_ c: NFKMLXDACConfiguration) {
        _convIn.wrappedValue = Conv1d(inputChannels: c.latentDim, outputChannels: c.decoderDim, kernelSize: 7, padding: 3)
        _blocks.wrappedValue = c.decoderRates.enumerated().map { index, stride in
            NFKMusic3VocoderBlock(inputDim: c.decoderDim >> index, outputDim: c.decoderDim >> (index + 1), stride: stride)
        }
        let outputDim = c.decoderDim >> c.decoderRates.count
        snake = NFKMusic3Snake(channels: outputDim)
        _convOut.wrappedValue = Conv1d(inputChannels: outputDim, outputChannels: 1, kernelSize: 7, padding: 3)
    }

    /// Quantized latent `[B, frames, latentDim]` → waveform `[B, samples, 1]` in `-1...1`.
    func callAsFunction(_ z: MLXArray) -> MLXArray {
        var out = convIn(z)
        for block in blocks {
            out = block(out)
        }
        return tanh(convOut(snake(out)))
    }
}

/// One vector quantizer: project the latent to the codebook width, match each frame to its nearest
/// (L2-normalized) codebook entry, and project the chosen entry back. The nearest-neighbor search runs
/// on normalized vectors; the reconstruction uses the raw codebook entry, as the reference does.
final class NFKDACVectorQuantize: Module {
    @ModuleInfo(key: "in_proj") var inProj: Conv1d
    @ModuleInfo(key: "out_proj") var outProj: Conv1d
    @ModuleInfo(key: "codebook") var codebook: Embedding

    init(inputDim: Int, codebookSize: Int, codebookDim: Int) {
        _inProj.wrappedValue = Conv1d(inputChannels: inputDim, outputChannels: codebookDim, kernelSize: 1)
        _outProj.wrappedValue = Conv1d(inputChannels: codebookDim, outputChannels: inputDim, kernelSize: 1)
        _codebook.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: codebookDim)
    }

    /// `[B, T, inputDim]` → the codes `[B, T]` and the projected reconstruction `[B, T, inputDim]`.
    func callAsFunction(_ z: MLXArray) -> (codes: MLXArray, reconstruction: MLXArray) {
        let latents = inProj(z)                                              // [B, T, codebookDim]
        let indices = Self.nearest(latents, codebook: codebook.weight)       // [B, T]
        return (indices, outProj(codebook(indices)))
    }

    /// Reconstructs `[B, T, inputDim]` from codes `[B, T]`.
    func decode(_ codes: MLXArray) -> MLXArray {
        outProj(codebook(codes))
    }

    /// The index of the nearest codebook entry per frame, comparing L2-normalized vectors. Maximizing
    /// the dot product of unit vectors is minimizing their Euclidean distance.
    static func nearest(_ latents: MLXArray, codebook weight: MLXArray) -> MLXArray {
        let normalized = latents / sqrt(latents.square().sum(axis: -1, keepDims: true) + 1e-12)
        let codes = weight / sqrt(weight.square().sum(axis: -1, keepDims: true) + 1e-12)
        let similarity = normalized.matmul(codes.transposed(1, 0))          // [B, T, codebookSize]
        return argMax(similarity, axis: -1)
    }
}

/// The residual vector quantizer: a stack of quantizers, each coding the residual the previous ones left.
final class NFKDACResidualVectorQuantize: Module {
    @ModuleInfo(key: "quantizers") var quantizers: [NFKDACVectorQuantize]

    init(_ c: NFKMLXDACConfiguration) {
        _quantizers.wrappedValue = (0 ..< c.codebooks).map { _ in
            NFKDACVectorQuantize(inputDim: c.latentDim, codebookSize: c.codebookSize, codebookDim: c.codebookDim)
        }
    }

    /// Latent `[B, T, latentDim]` → codes `[B, codebooks, T]`.
    func encode(_ z: MLXArray) -> MLXArray {
        var residual = z
        var codes = [MLXArray]()
        for quantizer in quantizers {
            let (indices, reconstruction) = quantizer(residual)
            residual = residual - reconstruction
            codes.append(indices)                                           // [B, T]
        }
        return stacked(codes, axis: 1)                                      // [B, codebooks, T]
    }

    /// Codes `[B, codebooks, T]` → the summed quantized latent `[B, T, latentDim]`.
    func decode(_ codes: MLXArray) -> MLXArray {
        var z = quantizers[0].decode(codes[0..., 0, 0...])
        for index in 1 ..< quantizers.count {
            z = z + quantizers[index].decode(codes[0..., index, 0...])
        }
        return z
    }
}

/// The DAC codec network: encoder, residual vector quantizer, and decoder.
final class NFKMLXDACNet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKDACEncoderNet
    @ModuleInfo(key: "quantizer") var quantizer: NFKDACResidualVectorQuantize
    @ModuleInfo(key: "decoder") var decoder: NFKDACDecoderNet

    let configuration: NFKMLXDACConfiguration

    init(_ c: NFKMLXDACConfiguration) {
        configuration = c
        _encoder.wrappedValue = NFKDACEncoderNet(c)
        _quantizer.wrappedValue = NFKDACResidualVectorQuantize(c)
        _decoder.wrappedValue = NFKDACDecoderNet(c)
    }

    /// A mono waveform → codes `[1, codebooks, frames]`. The clip is padded to a whole number of hops.
    func encode(_ samples: [Float]) -> MLXArray {
        let hop = configuration.hopLength
        var signal = samples
        let remainder = signal.count % hop
        if remainder != 0 {
            signal += [Float](repeating: 0, count: hop - remainder)
        }
        let audio = signal.withUnsafeBufferPointer { MLXArray($0, [1, signal.count, 1]) }
        return quantizer.encode(encoder(audio))
    }

    /// Codes `[1, codebooks, frames]` → a mono waveform `[samples]` in `-1...1`.
    func decode(_ codes: MLXArray) -> MLXArray {
        let wave = decoder(quantizer.decode(codes))                         // [1, samples, 1]
        eval(wave)
        return wave.reshaped([-1])
    }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKDACHolder: @unchecked Sendable {
    let net: NFKMLXDACNet
    init(_ net: NFKMLXDACNet) { self.net = net }
}

/// DAC reconstruction (audio → codes → audio) as an InferKit backend, reading `NFKInputAudio` and
/// returning the reconstructed clip under `NFKOutputAudio`. The codes themselves are reached through
/// `NFKMLXDAC.encode`/`decode`, which is what a codec-token consumer wants.
@objc(NFKMLXDACBackend)
public final class NFKMLXDACBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKDACHolder
    private let identifier: String

    init(net: NFKMLXDACNet, identifier: String) {
        holder = NFKDACHolder(net)
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
        let waveform = holder.net.decode(holder.net.encode(matched)).asArray(Float.self)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("dac-\(UUID().uuidString).wav")
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

/// Registration, encode/decode, and weight loading for the Descript Audio Codec.
@objc(NFKMLXDAC)
public final class NFKMLXDAC: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "dac"

    private let holder: NFKDACHolder

    init(net: NFKMLXDACNet) { holder = NFKDACHolder(net) }

    static func makeNet(_ configuration: NFKMLXDACConfiguration = .dac44kHz) -> NFKMLXDACNet {
        NFKMLXDACNet(configuration)
    }

    /// Encodes a mono waveform to its codebook token streams: `codes[c][t]` is codebook `c`'s token at
    /// frame `t`. This is the codec's product for a codec-token consumer.
    public func encode(_ samples: [Float]) -> [[Int]] {
        let codes = holder.net.encode(samples)                              // [1, codebooks, frames]
        eval(codes)
        let (books, frames) = (codes.shape[1], codes.shape[2])
        let flat = codes.reshaped([books, frames]).asType(.int32).asArray(Int32.self)
        return (0 ..< books).map { book in (0 ..< frames).map { Int(flat[book * frames + $0]) } }
    }

    /// Reconstructs a mono waveform from codebook token streams.
    public func decode(_ codes: [[Int]]) -> [Float] {
        let books = codes.count, frames = codes.first?.count ?? 0
        let flat = codes.flatMap { $0.map(Int32.init) }
        let array = flat.withUnsafeBufferPointer { MLXArray($0, [1, books, frames]) }
        return holder.net.decode(array).asArray(Float.self)
    }

    /// Builds a DAC codec directly from optional local weights — no registry required. A nil
    /// `weightsURL` builds random weights (`isReady` is true). Run off the render thread.
    ///
    /// - Since: InferKit 0.3.1
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        NFKMLXDACBackend(net: try loadedNet(.dac44kHz, weightsURL: weightsURL), identifier: modelName)
    }

    /// Builds a DAC codec object (with `encode`/`decode`) directly from optional local weights.
    public static func codec(configuration: NFKMLXDACConfiguration = .dac44kHz, weightsURL: URL?) throws -> NFKMLXDAC {
        NFKMLXDAC(net: try loadedNet(configuration, weightsURL: weightsURL))
    }

    private static func loadedNet(_ configuration: NFKMLXDACConfiguration, weightsURL: URL?) throws -> NFKMLXDACNet {
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

    /// Registers DAC (`dac`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a checkpoint, fusing weight-normalized convolutions (`g·v/‖v‖`) and transposing Conv1d
    /// weights to MLX's layout, reusing the Music 3 vocoder's weight loader. The reference keeps the
    /// encoder, quantizer, and decoder as flat `nn.Sequential` lists; `remapReferenceKey` names them.
    static func loadWeights(into net: NFKMLXDACNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        // A fine-tuned save is already in the module's layout and fused: apply with no transpose.
        guard checkpoint.needsConvTranspose else {
            try NFKMLXWeights.apply(checkpoint.arrays.map { (remapReferenceKey($0, net.configuration), $1) }, to: net)
            return
        }
        let mapped = NFKMLXMusic3.fusedWeightNorm(checkpoint.arrays).map { key, value -> (String, MLXArray) in
            let name = remapReferenceKey(key, net.configuration)
            if key.hasSuffix(".alpha") {                                 // Snake α: [1, C, 1] → [1, 1, C]
                return (name, value.transposed(0, 2, 1))
            }
            if name.contains("conv_t1"), value.ndim == 3 {               // ConvT [in, out, K] → [out, K, 1, in]
                return (name, value.transposed(1, 2, 0).expandedDimensions(axis: 2))
            }
            if value.ndim == 3 {                                         // Conv [out, in, K] → [out, K, in]
                return (name, value.transposed(0, 2, 1))
            }
            return (name, value)                                         // codebook (2-D) and biases pass through
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Maps the reference's nested positional `nn.Sequential` names onto the module's names. The encoder
    /// is `encoder.block.N` (0 the first convolution, 1…B the downsampling blocks, B+1 the Snake, B+2 the
    /// projection); each block is `block.M` (0…2 the residual units, 3 the Snake, 4 the strided
    /// convolution); each residual unit is `block.K` (0 Snake, 1 the dilated conv, 2 Snake, 3 the 1×1
    /// conv). The decoder mirrors it: `decoder.model.N` (0 the projection, 1…B the upsampling blocks, B+1
    /// the Snake, B+2 the output conv), each block `block.M` (0 Snake, 1 the transposed conv, 2…4 the
    /// residual units). The quantizer keys already match the module names.
    static func remapReferenceKey(_ key: String, _ c: NFKMLXDACConfiguration) -> String {
        let parts = key.split(separator: ".").map(String.init)
        if parts.first == "quantizer" { return key }

        // A residual unit's inner Sequential: 0 → snake1, 1 → conv1, 2 → snake2, 3 → conv2.
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
            if n == blocks + 1 { return "encoder.snake." + parts[3...].joined(separator: ".") }
            if n == blocks + 2 { return "encoder.conv_out." + parts[3...].joined(separator: ".") }
            guard parts.count >= 5, parts[3] == "block", let m = Int(parts[4]) else { return key }
            let prefix = "encoder.blocks.\(n - 1)"
            if m <= 2 { return "\(prefix).res_unit\(m + 1)." + residualUnit(Array(parts[5...])) }
            if m == 3 { return "\(prefix).snake." + parts[5...].joined(separator: ".") }
            return "\(prefix).conv." + parts[5...].joined(separator: ".")
        }

        if parts.first == "decoder", parts.count >= 4, parts[1] == "model", let n = Int(parts[2]) {
            let blocks = c.decoderRates.count
            if n == 0 { return "decoder.conv_in." + parts[3...].joined(separator: ".") }
            if n == blocks + 1 { return "decoder.snake." + parts[3...].joined(separator: ".") }
            if n == blocks + 2 { return "decoder.conv_out." + parts[3...].joined(separator: ".") }
            guard parts.count >= 5, parts[3] == "block", let m = Int(parts[4]) else { return key }
            let prefix = "decoder.blocks.\(n - 1)"
            if m == 0 { return "\(prefix).snake1." + parts[5...].joined(separator: ".") }
            if m == 1 { return "\(prefix).conv_t1.conv." + parts[5...].joined(separator: ".") }
            return "\(prefix).res_unit\(m - 1)." + residualUnit(Array(parts[5...]))
        }
        return key
    }
}
