//
//  NFKMLXNUWave2.swift
//  InferKitMLX
//
//  NU-Wave 2 (maum-ai/nuwave2, BSD-3): diffusion bandwidth extension. A WaveGrad-style noise
//  predictor whose residual blocks are short-time Fourier convolutions (a local convolution branch
//  beside an STFT-domain 1×1 convolution modulated by the input's bandwidth), sampled by an eight-step
//  logSNR DDIM from standard-normal noise conditioned on the upsampled narrow-band clip.
//

import Foundation
import InferKit
import MLX
import MLXFFT
import MLXNN

/// The released `hparameter.yaml` geometry.
public struct NFKMLXNUWave2Configuration: Sendable {
    public var sampleRate = 48_000
    public var fftSize = 1024
    public var hopSize = 256
    public var residualLayers = 15
    public var residualChannels = 64
    public var positionEmbeddingDimension = 512
    public var bsftChannels = 64
    public var positionEmbeddingScale: Float = 50_000
    public var positionEmbeddingChannels = 128
    public var logSNRMinimum: Float = -20
    public var logSNRMaximum: Float = 20
    /// `infer_schedule`, the eight logSNR values the released sampler visits.
    public var schedule: [Float] = [-2.6, -0.8, 2.0, 6.4, 9.8, 12.9, 14.4, 17.2]
    public var bins: Int { fftSize / 2 + 1 }
    public init() {}
}

// MARK: - The normalized STFT pair

/// `torch.stft` / `torch.istft` with `center=true` (reflect), a periodic Hann of the FFT size, over a
/// batch of channels `[N, L]`, the hop dividing the FFT size. `normalized` scales both transforms by
/// `1/√nFFT` and its inverse. The overlap-add reshapes each frame into `fftSize / hop` chunks and sums
/// the shifted chunk sequences (MLX has no scatter-add). Apollo shares it un-normalized.
struct NFKNUWaveSpectrum {
    let nFFT: Int
    let hop: Int
    let normalized: Bool
    let window: MLXArray                                                  // [nFFT]

    init(nFFT: Int, hop: Int, normalized: Bool = true) {
        self.nFFT = nFFT
        self.hop = hop
        self.normalized = normalized
        window = MLXArray(nfkPeriodicHann(nFFT))
    }

    /// `[N, L]` → real and imaginary `[N, frames, bins]`.
    func transform(_ signal: MLXArray) -> (real: MLXArray, imaginary: MLXArray) {
        let length = signal.dim(1)
        let pad = nFFT / 2
        var indices = [Int32]()
        for i in stride(from: pad, through: 1, by: -1) { indices.append(Int32(i)) }
        indices.append(contentsOf: (0 ..< length).map { Int32($0) })
        for i in stride(from: length - 2, through: length - 1 - pad, by: -1) { indices.append(Int32(i)) }
        let padded = take(signal, MLXArray(indices), axis: 1)
        let frames = 1 + (length + 2 * pad - nFFT) / hop
        var gather = [Int32]()
        gather.reserveCapacity(frames * nFFT)
        for f in 0 ..< frames { for k in 0 ..< nFFT { gather.append(Int32(f * hop + k)) } }
        let framed = take(padded, MLXArray(gather), axis: 1).reshaped([signal.dim(0), frames, nFFT]) * window.reshaped([1, 1, nFFT])
        var spectrum = MLXFFT.rfft(framed, axis: 2)
        if normalized { spectrum = spectrum / Float(nFFT).squareRoot() }
        return (spectrum.realPart(), spectrum.imaginaryPart())
    }

    /// Real and imaginary `[N, frames, bins]` → `[N, L]`, `L = (frames − 1) · hop` unless `length` is
    /// given (`torch.istft`'s `length`: the output runs that many samples from the center pad).
    func inverse(real: MLXArray, imaginary: MLXArray, length requested: Int? = nil) -> MLXArray {
        let (n, frames) = (real.dim(0), real.dim(1))
        var complex = real.asType(.complex64) + imaginary.asType(.complex64) * MLXArray(real: 0, imaginary: 1)
        if normalized { complex = complex * Float(nFFT).squareRoot() }
        let time = MLXFFT.irfft(complex, n: nFFT, axis: 2) * window.reshaped([1, 1, nFFT])
        let chunks = nFFT / hop
        let total = (frames - 1) * hop + nFFT
        func overlapAdd(_ frameValues: MLXArray) -> MLXArray {                // [N', frames, nFFT] → [N', total]
            let split = frameValues.reshaped([frameValues.dim(0), frames, chunks, hop])
            var sum: MLXArray? = nil
            for k in 0 ..< chunks {
                let sequence = split[0..., 0..., k, 0...].reshaped([frameValues.dim(0), frames * hop])
                let shifted = MLX.padded(sequence, widths: [IntOrPair(0), IntOrPair((k * hop, total - frames * hop - k * hop))], mode: .constant)
                sum = sum.map { $0 + shifted } ?? shifted
            }
            return sum!
        }
        let output = overlapAdd(time)
        let envelope = overlapAdd(broadcast(window.square().reshaped([1, 1, nFFT]), to: [1, frames, nFFT]))
        let length = requested ?? (frames - 1) * hop
        let pad = nFFT / 2
        let span = envelope[0..., pad ..< pad + length]
        let interior = MLX.where(span .> 1e-11, output[0..., pad ..< pad + length] / span, MLXArray(Float(0)))
        return interior.reshaped([n, length])
    }
}

// MARK: - Modules

/// `DiffusionEmbedding`: a scaled sinusoidal embedding of the normalized negative logSNR, two SiLU
/// projections.
final class NFKNUWaveDiffusionEmbedding: Module {
    @ModuleInfo(key: "projection1") var projection1: Linear
    @ModuleInfo(key: "projection2") var projection2: Linear
    let channels: Int
    let scale: Float

    init(_ c: NFKMLXNUWave2Configuration) {
        channels = c.positionEmbeddingChannels
        scale = c.positionEmbeddingScale
        _projection1.wrappedValue = Linear(c.positionEmbeddingChannels, c.positionEmbeddingDimension)
        _projection2.wrappedValue = Linear(c.positionEmbeddingDimension, c.positionEmbeddingDimension)
    }

    /// `[B]` → `[B, dimension]`.
    func callAsFunction(_ level: MLXArray) -> MLXArray {
        let half = channels / 2
        let frequencies = exp(MLXArray(0 ..< Int32(half)).asType(.float32) * (-Float(log(10000.0)) / Float(half - 1)))
        let angles = scale * level.reshaped([-1, 1]) * frequencies.reshaped([1, half])
        let embedding = concatenated([sin(angles), cos(angles)], axis: 1)
        return silu(projection2(silu(projection1(embedding))))
    }
}

/// `BSFT`: the bandwidth one-hot `[B, bins, 2]` through a shared 3-tap convolution over the bins to a
/// per-bin scale and shift for the spectrum's channels.
final class NFKNUWaveBSFT: Module {
    @ModuleInfo(key: "mlp_shared") var shared: Conv1d
    @ModuleInfo(key: "mlp_gamma") var gamma: Conv1d
    @ModuleInfo(key: "mlp_beta") var beta: Conv1d

    init(hidden: Int, channels: Int) {
        _shared.wrappedValue = Conv1d(inputChannels: 2, outputChannels: hidden, kernelSize: 3, padding: 1)
        _gamma.wrappedValue = Conv1d(inputChannels: hidden, outputChannels: channels, kernelSize: 3, padding: 1)
        _beta.wrappedValue = Conv1d(inputChannels: hidden, outputChannels: channels, kernelSize: 3, padding: 1)
    }

    /// `x [B, bins, frames, channels]`, `band [B, bins, 2]` → `x · (1 + γ) + β`.
    func callAsFunction(_ x: MLXArray, band: MLXArray) -> MLXArray {
        let activated = silu(shared(band))
        let g = gamma(activated).expandedDimensions(axis: 2)
        let b = beta(activated).expandedDimensions(axis: 2)
        return x * (1 + g) + b
    }
}

/// `FourierUnit`: every channel to its normalized STFT, the real and imaginary parts interleaved on
/// the channel axis (`2c`, `2c + 1`), BSFT, ReLU, a bias-free 1×1 convolution across those channels,
/// and the inverse STFT.
final class NFKNUWaveFourierUnit: Module {
    @ModuleInfo(key: "conv_layer") var conv: Conv2d
    @ModuleInfo(key: "bsft") var bsft: NFKNUWaveBSFT
    let spectrum: NFKNUWaveSpectrum

    init(channels: Int, bsftChannels: Int, spectrum: NFKNUWaveSpectrum) {
        self.spectrum = spectrum
        _conv.wrappedValue = Conv2d(inputChannels: channels * 2, outputChannels: channels * 2, kernelSize: 1, bias: false)
        _bsft.wrappedValue = NFKNUWaveBSFT(hidden: bsftChannels, channels: channels * 2)
    }

    /// `x [B, L, C]`, `band [B, bins, 2]` → `[B, L, C]`.
    func callAsFunction(_ x: MLXArray, band: MLXArray) -> MLXArray {
        let (b, length, c) = (x.dim(0), x.dim(1), x.dim(2))
        let signals = x.transposed(0, 2, 1).reshaped([b * c, length])
        let (real, imaginary) = spectrum.transform(signals)                   // [B·C, F, bins]
        let (frames, bins) = (real.dim(1), real.dim(2))
        var packed = stacked([real.reshaped([b, c, frames, bins]), imaginary.reshaped([b, c, frames, bins])], axis: -1)
        packed = packed.transposed(0, 3, 2, 1, 4).reshaped([b, bins, frames, c * 2])   // channel = 2c + part
        let mixed = conv(relu(bsft(packed, band: band)))
        let unpacked = mixed.reshaped([b, bins, frames, c, 2]).transposed(0, 3, 2, 1, 4)   // [B, C, F, bins, 2]
        let outReal = unpacked[0..., 0..., 0..., 0..., 0].reshaped([b * c, frames, bins])
        let outImaginary = unpacked[0..., 0..., 0..., 0..., 1].reshaped([b * c, frames, bins])
        return spectrum.inverse(real: outReal, imaginary: outImaginary).reshaped([b, c, length]).transposed(0, 2, 1)
    }
}

/// `SpectralTransform`: a 1×1 convolution and SiLU, the Fourier unit, and a 1×1 convolution over the
/// sum of the two.
final class NFKNUWaveSpectralTransform: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "fu") var fourier: NFKNUWaveFourierUnit
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    init(input: Int, output: Int, bsftChannels: Int, spectrum: NFKNUWaveSpectrum) {
        _conv1.wrappedValue = Conv1d(inputChannels: input, outputChannels: output / 2, kernelSize: 1, bias: false)
        _fourier.wrappedValue = NFKNUWaveFourierUnit(channels: output / 2, bsftChannels: bsftChannels, spectrum: spectrum)
        _conv2.wrappedValue = Conv1d(inputChannels: output / 2, outputChannels: output, kernelSize: 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray, band: MLXArray) -> MLXArray {
        let h = silu(conv1(x))
        return conv2(h + fourier(h, band: band))
    }
}

/// `FFC` (the short-time Fourier convolution): local and global halves cross-connected through three
/// 3-tap convolutions and the spectral transform.
final class NFKNUWaveFFC: Module {
    @ModuleInfo(key: "convl2l") var localToLocal: Conv1d
    @ModuleInfo(key: "convl2g") var localToGlobal: Conv1d
    @ModuleInfo(key: "convg2l") var globalToLocal: Conv1d
    @ModuleInfo(key: "convg2g") var globalToGlobal: NFKNUWaveSpectralTransform
    let globalInputs: Int

    init(input: Int, output: Int, bsftChannels: Int, spectrum: NFKNUWaveSpectrum) {
        let inGlobal = input / 2, inLocal = input - inGlobal
        let outGlobal = output / 2, outLocal = output - outGlobal
        globalInputs = inGlobal
        _localToLocal.wrappedValue = Conv1d(inputChannels: inLocal, outputChannels: outLocal, kernelSize: 3, padding: 1, bias: false)
        _localToGlobal.wrappedValue = Conv1d(inputChannels: inLocal, outputChannels: outGlobal, kernelSize: 3, padding: 1, bias: false)
        _globalToLocal.wrappedValue = Conv1d(inputChannels: inGlobal, outputChannels: outLocal, kernelSize: 3, padding: 1, bias: false)
        _globalToGlobal.wrappedValue = NFKNUWaveSpectralTransform(input: inGlobal, output: outGlobal, bsftChannels: bsftChannels, spectrum: spectrum)
    }

    func callAsFunction(local: MLXArray, global: MLXArray, band: MLXArray) -> (local: MLXArray, global: MLXArray) {
        (localToLocal(local) + globalToLocal(global), localToGlobal(local) + globalToGlobal(global, band: band))
    }
}

/// `ResidualBlock`: the noise level projected onto the channels, the FFC over the channel halves, a
/// gated activation whose gate and filter halves are gathered from both branches, and a 1×1
/// projection splitting into the residual (scaled by `1/√2`) and the skip.
final class NFKNUWaveResidualBlock: Module {
    @ModuleInfo(key: "ffc1") var ffc: NFKNUWaveFFC
    @ModuleInfo(key: "diffusion_projection") var diffusionProjection: Linear
    @ModuleInfo(key: "output_projection") var outputProjection: Conv1d

    init(_ c: NFKMLXNUWave2Configuration, spectrum: NFKNUWaveSpectrum) {
        let width = c.residualChannels
        _ffc.wrappedValue = NFKNUWaveFFC(input: width, output: 2 * width, bsftChannels: c.bsftChannels, spectrum: spectrum)
        _diffusionProjection.wrappedValue = Linear(c.positionEmbeddingDimension, width)
        _outputProjection.wrappedValue = Conv1d(inputChannels: width, outputChannels: 2 * width, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray, band: MLXArray, level: MLXArray) -> (residual: MLXArray, skip: MLXArray) {
        let width = x.dim(2)
        let y = x + diffusionProjection(level).expandedDimensions(axis: 1)
        let split = width - ffc.globalInputs
        let (local, global) = ffc(local: y[0..., 0..., ..<split], global: y[0..., 0..., split...], band: band)
        let half = local.dim(2) / 2
        let gate = concatenated([local[0..., 0..., ..<half], global[0..., 0..., ..<half]], axis: 2)
        let filter = concatenated([local[0..., 0..., half...], global[0..., 0..., half...]], axis: 2)
        let projected = outputProjection(sigmoid(gate) * tanh(filter))
        let residual = projected[0..., 0..., ..<width], skip = projected[0..., 0..., width...]
        return ((x + residual) / Float(2).squareRoot(), skip)
    }
}

/// The NU-Wave 2 noise predictor `NuWave2`: `[noisy, narrow-band]` → a 1×1 projection and SiLU, the
/// residual blocks with summed skips, `skip / √layers`, a 1×1 projection, SiLU, and the output
/// projection to one channel.
public final class NFKMLXNUWave2Net: Module {
    @ModuleInfo(key: "input_projection") var inputProjection: Conv1d
    @ModuleInfo(key: "diffusion_embedding") var embedding: NFKNUWaveDiffusionEmbedding
    @ModuleInfo(key: "residual_layers") var layers: [NFKNUWaveResidualBlock]
    @ModuleInfo(key: "skip_projection") var skipProjection: Conv1d
    @ModuleInfo(key: "output_projection") var outputProjection: Conv1d
    let configuration: NFKMLXNUWave2Configuration
    let spectrum: NFKNUWaveSpectrum

    init(_ c: NFKMLXNUWave2Configuration) {
        configuration = c
        spectrum = NFKNUWaveSpectrum(nFFT: c.fftSize, hop: c.hopSize)
        _inputProjection.wrappedValue = Conv1d(inputChannels: 2, outputChannels: c.residualChannels, kernelSize: 1)
        _embedding.wrappedValue = NFKNUWaveDiffusionEmbedding(c)
        let shared = spectrum
        _layers.wrappedValue = (0 ..< c.residualLayers).map { _ in NFKNUWaveResidualBlock(c, spectrum: shared) }
        _skipProjection.wrappedValue = Conv1d(inputChannels: c.residualChannels, outputChannels: c.residualChannels, kernelSize: 1)
        _outputProjection.wrappedValue = Conv1d(inputChannels: c.residualChannels, outputChannels: 1, kernelSize: 1)
        super.init()
    }

    /// `F.one_hot(band)` as `[B, bins, 2]` from a `[B, bins]` 0/1 band.
    static func oneHot(_ band: MLXArray) -> MLXArray {
        let ones = band.asType(.float32)
        return stacked([1 - ones, ones], axis: -1)
    }

    /// `audio`, `narrowband` `[B, L]`, `band [B, bins]` (0/1), `level [B]` → the predicted noise `[B, L]`.
    public func callAsFunction(_ audio: MLXArray, narrowband: MLXArray, band: MLXArray, level: MLXArray) -> MLXArray {
        let (states, _) = stages(audio, narrowband: narrowband, band: band, level: level)
        return states
    }

    /// The forward with the first block's outputs exposed for the parity harness.
    func stages(_ audio: MLXArray, narrowband: MLXArray, band: MLXArray, level: MLXArray)
        -> (output: MLXArray, firstBlock: (residual: MLXArray, skip: MLXArray)) {
        var x = silu(inputProjection(stacked([audio, narrowband], axis: -1)))     // [B, L, C]
        let embedded = embedding(level)
        let oneHotBand = Self.oneHot(band)
        var skip: MLXArray? = nil
        var first: (MLXArray, MLXArray)? = nil
        for layer in layers {
            let (residual, layerSkip) = layer(x, band: oneHotBand, level: embedded)
            x = residual
            if first == nil { first = (residual, layerSkip) }
            skip = skip.map { $0 + layerSkip } ?? layerSkip
        }
        let summed = skip! / Float(layers.count).squareRoot()
        let output = outputProjection(silu(skipProjection(summed)))
        return (output.squeezed(axis: 2), first!)
    }
}

// MARK: - Sampler

/// `Diffusion.denoise_ddim` over `infer_schedule`: at each logSNR `t` the network predicts the noise
/// of the current signal, the clean estimate is `(y − σ_t · ε) / α_t`, and the next signal is
/// `α_s · x̂ + σ_s · ε` with `α² = sigmoid(logSNR)`, `σ² = sigmoid(−logSNR)`, the last step landing
/// on `logsnr_max`. Any other step count walks the logSNR range evenly.
enum NFKMLXNUWave2Sampler {
    static func sample(_ net: NFKMLXNUWave2Net, narrowband: MLXArray, band: MLXArray, noise: MLXArray,
                       steps: Int? = nil) -> (final: MLXArray, trajectory: [MLXArray]) {
        let c = net.configuration
        let schedule: [Float]
        let count = steps ?? c.schedule.count
        if count == c.schedule.count {
            schedule = c.schedule
        } else {
            let h = (c.logSNRMaximum - c.logSNRMinimum) / Float(count)
            schedule = (0 ..< count).map { c.logSNRMinimum + Float($0) * h }
        }
        var signal = noise
        var trajectory = [MLXArray]()
        for i in 0 ..< count {
            let t = schedule[i]
            let s = i == count - 1 ? c.logSNRMaximum : (count == c.schedule.count ? schedule[i + 1] : c.logSNRMinimum + Float(i + 1) * (c.logSNRMaximum - c.logSNRMinimum) / Float(count))
            let level = (c.logSNRMaximum - t) / (c.logSNRMaximum - c.logSNRMinimum)
            let predicted = net(signal, narrowband: narrowband, band: band, level: MLXArray([level]))
            let alphaT = sigmoid(MLXArray(t)).sqrt(), sigmaT = sigmoid(MLXArray(-t)).sqrt()
            let alphaS = sigmoid(MLXArray(s)).sqrt(), sigmaS = sigmoid(MLXArray(-s)).sqrt()
            let clean = (signal - sigmaT * predicted) / alphaT
            signal = alphaS * clean + sigmaS * predicted
            eval(signal)
            trajectory.append(signal)
        }
        // torch.clamp(-1, 1 - finfo(float16).eps)
        return (clip(signal, min: -1, max: 1 - 0.0009765625), trajectory)
    }
}

// MARK: - Backend

private final class NFKNUWaveHolder: @unchecked Sendable {
    let net: NFKMLXNUWave2Net
    init(_ net: NFKMLXNUWave2Net) { self.net = net }
}

/// NU-Wave 2 as an InferKit backend: a narrow-band `NFKInputAudio` at its own rate → the 48 kHz
/// bandwidth-extended clip under `NFKOutputAudio`. `NFKParameterSeed` fixes the diffusion start.
@objc(NFKMLXNUWave2Backend)
public final class NFKMLXNUWave2Backend: NSObject, NFKInferenceBackend {
    private let holder: NFKNUWaveHolder
    private let identifier: String
    private let outputDirectory: URL

    init(net: NFKMLXNUWave2Net, identifier: String, outputDirectory: URL = FileManager.default.temporaryDirectory) {
        holder = NFKNUWaveHolder(net)
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
        let seed = (request.parameter(forKey: NFKParameterSeed) as? NSNumber)?.uint64Value ?? 0
        let enhanced = Self.enhance(samples, sampleRate: rate, net: holder.net, seed: seed)
        let url = outputDirectory.appendingPathComponent("nuwave2-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: enhanced, sampleRate: configuration.sampleRate, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(enhanced.count) / Double(configuration.sampleRate),
                                  sampleRate: Double(configuration.sampleRate), channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
    }

    /// `inference.py`'s conditioning: the clip peak-normalized, upsampled to 48 kHz, trimmed to a
    /// multiple of the hop; the band marks the first `int((rate / 2) / 24000 · bins)` bins.
    static func prepared(_ samples: [Float], sampleRate: Int, configuration c: NFKMLXNUWave2Configuration) -> (narrowband: [Float], band: [Int32]) {
        let peak = samples.map { abs($0) }.max() ?? 1
        let normalized = samples.map { $0 / (peak > 0 ? peak : 1) }
        var upsampled = sampleRate == c.sampleRate ? normalized : NFKMLXAudioRate.matched(normalized, from: sampleRate, to: c.sampleRate)
        upsampled = Array(upsampled.prefix(upsampled.count - upsampled.count % c.hopSize))
        let cutoff = Int((Double(sampleRate / 2) / (0.5 * Double(c.sampleRate))) * Double(c.bins))
        return (upsampled, (0 ..< c.bins).map { $0 < cutoff ? 1 : 0 })
    }

    /// The whole path from a narrow-band clip at `sampleRate`.
    static func enhance(_ samples: [Float], sampleRate: Int, net: NFKMLXNUWave2Net, seed: UInt64) -> [Float] {
        let c = net.configuration
        let (narrowband, band) = prepared(samples, sampleRate: sampleRate, configuration: c)
        let low = narrowband.withUnsafeBufferPointer { MLXArray($0, [1, narrowband.count]) }
        let noise = MLXRandom.normal([1, narrowband.count], key: MLXRandom.key(seed))
        let (final, _) = NFKMLXNUWave2Sampler.sample(net, narrowband: low, band: MLXArray(band).reshaped([1, c.bins]), noise: noise)
        eval(final)
        return final.reshaped([-1]).asArray(Float.self)
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

/// Registration and weight loading for NU-Wave 2.
@objc(NFKMLXNUWave2)
public final class NFKMLXNUWave2: NSObject {
    @objc public static let modelName = "nuwave2"

    static func makeNet(_ configuration: NFKMLXNUWave2Configuration = .init()) -> NFKMLXNUWave2Net {
        NFKMLXNUWave2Net(configuration)
    }

    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL { try loadWeights(into: net, from: weightsURL) }
        return NFKMLXNUWave2Backend(net: net, identifier: modelName)
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

    /// Registers `nuwave2` with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads the official Lightning checkpoint (its `state_dict` through the native torch reader; the
    /// Lightning callbacks it pickles are inert). Keys sit under `model.model.` (the LightningModule's
    /// `Diffusion`, whose `NuWave2` is `model`); the STFT window buffers are dropped; the 1-D and 1×1
    /// 2-D convolutions transpose to channels-last.
    static func loadWeights(into net: NFKMLXNUWave2Net, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped: [(String, MLXArray)] = checkpoint.arrays.compactMap { key, value in
            guard let name = remapReferenceKey(key) else { return nil }
            var tensor = value
            if checkpoint.needsConvTranspose {
                if value.ndim == 4 { tensor = value.transposed(0, 2, 3, 1) }
                if value.ndim == 3 { tensor = value.transposed(0, 2, 1) }
            }
            return (name, tensor)
        }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }

    static func remapReferenceKey(_ key: String) -> String? {
        if key.hasSuffix("hann_window") { return nil }
        var name = key
        for prefix in ["model.model.", "model."] where name.hasPrefix(prefix) {
            name = String(name.dropFirst(prefix.count))
            break
        }
        return name
    }
}
