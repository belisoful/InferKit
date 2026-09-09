//
//  NFKMLXMossFormer2SR.swift
//  InferKitMLX
//
//  MossFormer2 SR 48K (modelscope/ClearerVoice-Studio, Apache-2.0): speech super-resolution as a
//  mel-to-mel MossFormer2 backbone over a HiFi-GAN log-mel, a Snake-activated HiFi-GAN generator, and
//  the reference decode path's bandwidth substitution (the input's own band kept, the generator's
//  band above it added).
//

import Foundation
import InferKit
import MLX
import MLXNN

/// The released MossFormer2 SR 48K geometry.
public struct NFKMLXMossFormer2SRConfiguration: Sendable {
    public var sampleRate = 48_000
    public var fftSize = 1024
    public var hopSize = 256
    public var numMels = 80
    public var melMaximumFrequency: Float = 8000
    public var upsampleRates = [8, 8, 2, 2]
    public var upsampleKernels = [16, 16, 4, 4]
    public var initialChannel = 1024
    public var resblockKernels = [3, 7, 11]
    public var resblockDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    /// `bandwidth_sub`'s cumulative-energy threshold, its detection STFT size, and its crossfade.
    public var bandwidthEnergyThreshold: Double = 0.9996
    public var bandwidthDetectionSize = 256
    public var transitionMilliseconds = 100
    public var backbone: NFKMLXMossFormer2Configuration = .superResolution
    public init() {}
}

// MARK: - Generator

/// `ResBlock1` with Snake activations: three (dilated conv, plain conv) pairs, each entered through its
/// own Snake, added back to the input.
final class NFKMossSRResBlock: Module {
    @ModuleInfo(key: "convs1") var convs1: [Conv1d]
    @ModuleInfo(key: "convs1_activates") var activates1: [NFKMusic3Snake]
    @ModuleInfo(key: "convs2") var convs2: [Conv1d]
    @ModuleInfo(key: "convs2_activates") var activates2: [NFKMusic3Snake]

    init(channels: Int, kernel: Int, dilations: [Int]) {
        _convs1.wrappedValue = dilations.map {
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel,
                   padding: (kernel * $0 - $0) / 2, dilation: $0)
        }
        _activates1.wrappedValue = dilations.map { _ in NFKMusic3Snake(channels: channels) }
        _convs2.wrappedValue = dilations.map { _ in
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel, padding: (kernel - 1) / 2)
        }
        _activates2.wrappedValue = dilations.map { _ in NFKMusic3Snake(channels: channels) }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = input
        for i in 0 ..< convs1.count {
            x = convs2[i](activates2[i](convs1[i](activates1[i](x)))) + x
        }
        return x
    }
}

/// The Snake HiFi-GAN generator: `conv_pre`, then per stage a Snake, a transposed-convolution
/// upsample, and the mean of three multi-receptive-field residual blocks; a final Snake, `conv_post`,
/// and `tanh`.
public final class NFKMLXMossFormer2SRGenerator: Module {
    @ModuleInfo(key: "conv_pre") var convPre: Conv1d
    @ModuleInfo(key: "snakes") var snakes: [NFKMusic3Snake]
    @ModuleInfo(key: "ups") var ups: [ConvTransposed1d]
    @ModuleInfo(key: "resblocks") var resblocks: [NFKMossSRResBlock]
    @ModuleInfo(key: "snake_post") var snakePost: NFKMusic3Snake
    @ModuleInfo(key: "conv_post") var convPost: Conv1d
    let numKernels: Int
    let stages: Int

    init(_ c: NFKMLXMossFormer2SRConfiguration) {
        numKernels = c.resblockKernels.count
        stages = c.upsampleRates.count
        _convPre.wrappedValue = Conv1d(inputChannels: c.numMels, outputChannels: c.initialChannel, kernelSize: 7, padding: 3)
        var snakeList = [NFKMusic3Snake](), upList = [ConvTransposed1d](), blockList = [NFKMossSRResBlock]()
        for (i, (u, k)) in zip(c.upsampleRates, c.upsampleKernels).enumerated() {
            let inCh = c.initialChannel / (1 << i), outCh = c.initialChannel / (1 << (i + 1))
            snakeList.append(NFKMusic3Snake(channels: inCh))
            upList.append(ConvTransposed1d(inputChannels: inCh, outputChannels: outCh, kernelSize: k, stride: u, padding: (k - u) / 2))
            for (rk, rd) in zip(c.resblockKernels, c.resblockDilations) {
                blockList.append(NFKMossSRResBlock(channels: outCh, kernel: rk, dilations: rd))
            }
        }
        _snakes.wrappedValue = snakeList
        _ups.wrappedValue = upList
        _resblocks.wrappedValue = blockList
        let finalChannels = c.initialChannel / (1 << c.upsampleRates.count)
        _snakePost.wrappedValue = NFKMusic3Snake(channels: finalChannels)
        _convPost.wrappedValue = Conv1d(inputChannels: finalChannels, outputChannels: 1, kernelSize: 7, padding: 3)
        super.init()
    }

    /// `mel [B, T, numMels]` → waveform `[B, T · ∏rates, 1]`.
    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = convPre(mel)
        for i in 0 ..< stages {
            x = ups[i](snakes[i](x))
            var xs = resblocks[i * numKernels](x)
            for j in 1 ..< numKernels { xs = xs + resblocks[i * numKernels + j](x) }
            x = xs / Float(numKernels)
        }
        return tanh(convPost(snakePost(x)))
    }
}

// MARK: - Bandwidth substitution (the reference's scipy post-process)

/// `utils/bandwidth_sub.py` in double precision: detect the input's effective bandwidth from a
/// 256-point Hann STFT's cumulative energy, keep the input below that frequency (a fourth-order
/// Butterworth low-pass, zero-phase) and the generator's output above it (the matching high-pass),
/// then crossfade from the input to the sum over the first 100 ms.
enum NFKMossBandwidthSubstitution {
    private struct Complex {
        var re: Double, im: Double
        static func + (a: Complex, b: Complex) -> Complex { Complex(re: a.re + b.re, im: a.im + b.im) }
        static func - (a: Complex, b: Complex) -> Complex { Complex(re: a.re - b.re, im: a.im - b.im) }
        static func * (a: Complex, b: Complex) -> Complex { Complex(re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re) }
        static func / (a: Complex, b: Complex) -> Complex {
            let d = b.re * b.re + b.im * b.im
            return Complex(re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d)
        }
    }

    /// `scipy.signal.stft(x, fs)` defaults (a periodic Hann of `size`, hop `size / 2`, zero boundary
    /// padding of `size / 2`, the end padded to a whole hop) → the per-bin energy summed over time,
    /// then `detect_bandwidth`'s cumulative-energy crossing (`argmax(cumulative >= threshold)`).
    static func detectedBandwidth(_ samples: [Float], sampleRate: Int, size: Int, threshold: Double) -> Double {
        let hop = size / 2
        let half = size / 2
        var x = [Double](repeating: 0, count: half) + samples.map(Double.init) + [Double](repeating: 0, count: half)
        let extra = (-(x.count - size) % hop + hop) % hop
        x += [Double](repeating: 0, count: extra)
        let window = (0 ..< size).map { 0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(size)) }
        let frames = (x.count - size) / hop + 1
        let bins = size / 2 + 1
        var energy = [Double](repeating: 0, count: bins)
        for f in 0 ..< frames {
            let frame = (0 ..< size).map { x[f * hop + $0] * window[$0] }
            for k in 0 ..< bins {
                var re = 0.0, im = 0.0
                for n in 0 ..< size {
                    let angle = -2 * Double.pi * Double(k * n) / Double(size)
                    re += frame[n] * cos(angle)
                    im += frame[n] * sin(angle)
                }
                energy[k] += re * re + im * im
            }
        }
        let total = energy.reduce(0, +)
        var cumulative = 0.0
        for k in 0 ..< bins {
            cumulative += energy[k]
            if cumulative / total >= threshold { return Double(k) * Double(sampleRate) / Double(size) }
        }
        return Double(bins - 1) * Double(sampleRate) / Double(size)
    }

    /// `scipy.signal.butter(4, cutoff / nyquist, btype)` as `(b, a)`: the Butterworth prototype poles,
    /// the pre-warped low-pass or high-pass transform, the bilinear transform, and the polynomial
    /// coefficients.
    static func butterworth(order: Int, normalizedCutoff: Double, highpass: Bool) -> (b: [Double], a: [Double]) {
        let poles = (0 ..< order).map { i -> Complex in
            let m = Double(-order + 1 + 2 * i)
            let angle = Double.pi * m / (2 * Double(order))
            return Complex(re: -cos(angle), im: -sin(angle))
        }
        let warped = 4 * tan(Double.pi * normalizedCutoff / 2)
        var zeros = [Complex]()
        var transformed = [Complex]()
        var gain: Double
        if highpass {
            transformed = poles.map { Complex(re: warped, im: 0) / $0 }
            zeros = [Complex](repeating: Complex(re: 0, im: 0), count: order)
            var product = Complex(re: 1, im: 0)
            for p in poles { product = product * Complex(re: -p.re, im: -p.im) }
            gain = (Complex(re: 1, im: 0) / product).re
        } else {
            transformed = poles.map { $0 * Complex(re: warped, im: 0) }
            gain = pow(warped, Double(order))
        }
        // Bilinear transform at fs = 2 (scipy's digital design), the extra zeros landing at −1.
        let fs2 = Complex(re: 4, im: 0)
        let digitalPoles = transformed.map { (fs2 + $0) / (fs2 - $0) }
        var digitalZeros = zeros.map { (fs2 + $0) / (fs2 - $0) }
        digitalZeros += [Complex](repeating: Complex(re: -1, im: 0), count: order - zeros.count)
        var numerator = Complex(re: 1, im: 0), denominator = Complex(re: 1, im: 0)
        for z in zeros { numerator = numerator * (fs2 - z) }
        for p in transformed { denominator = denominator * (fs2 - p) }
        gain *= (numerator / denominator).re
        func polynomial(_ roots: [Complex]) -> [Double] {
            var coefficients = [Complex(re: 1, im: 0)]
            for r in roots {
                var next = [Complex](repeating: Complex(re: 0, im: 0), count: coefficients.count + 1)
                for (i, c) in coefficients.enumerated() {
                    next[i] = next[i] + c
                    next[i + 1] = next[i + 1] - c * r
                }
                coefficients = next
            }
            return coefficients.map(\.re)
        }
        return (polynomial(digitalZeros).map { $0 * gain }, polynomial(digitalPoles))
    }

    /// `scipy.signal.lfilter_zi`: the steady-state delay line for a unit step.
    private static func initialConditions(b: [Double], a: [Double]) -> [Double] {
        let n = max(a.count, b.count)
        let ap = a + [Double](repeating: 0, count: n - a.count)
        let bp = b + [Double](repeating: 0, count: n - b.count)
        // (I − companion(a)ᵀ) · zi = b[1:] − a[1:] · b[0]
        let m = n - 1
        var matrix = [[Double]](repeating: [Double](repeating: 0, count: m), count: m)
        for i in 0 ..< m {
            matrix[i][i] = 1
            matrix[i][0] += ap[i + 1]
            if i + 1 < m { matrix[i][i + 1] -= 1 }
        }
        var rhs = (0 ..< m).map { bp[$0 + 1] - ap[$0 + 1] * bp[0] }
        for column in 0 ..< m {
            var pivot = column
            for row in column + 1 ..< m where abs(matrix[row][column]) > abs(matrix[pivot][column]) { pivot = row }
            matrix.swapAt(column, pivot); rhs.swapAt(column, pivot)
            for row in 0 ..< m where row != column {
                let factor = matrix[row][column] / matrix[column][column]
                for k in column ..< m { matrix[row][k] -= factor * matrix[column][k] }
                rhs[row] -= factor * rhs[column]
            }
        }
        return (0 ..< m).map { rhs[$0] / matrix[$0][$0] }
    }

    /// Direct-form II transposed `lfilter` from an initial delay line.
    private static func filtered(_ x: [Double], b: [Double], a: [Double], initial: [Double]) -> [Double] {
        let n = max(a.count, b.count)
        let ap = a + [Double](repeating: 0, count: n - a.count)
        let bp = b + [Double](repeating: 0, count: n - b.count)
        var z = initial + [0]
        var y = [Double](repeating: 0, count: x.count)
        for i in 0 ..< x.count {
            let out = bp[0] * x[i] + z[0]
            for k in 0 ..< n - 1 { z[k] = bp[k + 1] * x[i] + z[k + 1] - ap[k + 1] * out }
            y[i] = out
        }
        return y
    }

    /// `scipy.signal.filtfilt` with its defaults: odd extension by `3 · max(len(a), len(b))`, a forward
    /// pass from the step-scaled initial state, a backward pass from the scaled final sample.
    static func zeroPhaseFiltered(_ x: [Double], b: [Double], a: [Double]) -> [Double] {
        let pad = 3 * max(a.count, b.count)
        let left = (1 ... pad).reversed().map { 2 * x[0] - x[$0] }
        let right = (0 ..< pad).map { 2 * x[x.count - 1] - x[x.count - 2 - $0] }
        let extended = left + x + right
        let zi = initialConditions(b: b, a: a)
        let forward = filtered(extended, b: b, a: a, initial: zi.map { $0 * extended[0] })
        let backward = Array(filtered(Array(forward.reversed()), b: b, a: a, initial: zi.map { $0 * forward[forward.count - 1] }).reversed())
        return Array(backward[pad ..< backward.count - pad])
    }

    /// `bandwidth_sub(low, high)`: the substituted signal at the shorter of the two lengths.
    static func substituted(low: [Float], high: [Float], configuration c: NFKMLXMossFormer2SRConfiguration) -> [Float] {
        let cutoff = detectedBandwidth(low, sampleRate: c.sampleRate, size: c.bandwidthDetectionSize,
                                       threshold: c.bandwidthEnergyThreshold)
        let normalized = cutoff / (0.5 * Double(c.sampleRate))
        let lowpass = butterworth(order: 4, normalizedCutoff: normalized, highpass: false)
        let highpass = butterworth(order: 4, normalizedCutoff: normalized, highpass: true)
        let kept = zeroPhaseFiltered(low.map(Double.init), b: lowpass.b, a: lowpass.a)
        let added = zeroPhaseFiltered(high.map(Double.init), b: highpass.b, a: highpass.a)
        let length = min(kept.count, added.count)
        let combined = (0 ..< length).map { kept[$0] + added[$0] }
        // smooth_transition: fade from the input to the substituted signal over the transition band.
        let fadeLength = c.transitionMilliseconds * c.sampleRate / 1000
        let count = min(combined.count, low.count)
        return (0 ..< count).map { i in
            let fade = i < fadeLength ? Double(i) / Double(fadeLength - 1) : 1
            return Float((1 - fade) * Double(low[i]) + fade * combined[i])
        }
    }
}

// MARK: - The model

/// MossFormer2 SR: the mel front end, the backbone, and the generator.
public final class NFKMLXMossFormer2SRNet {
    public let backbone: NFKMLXMossFormer2SENet
    public let generator: NFKMLXMossFormer2SRGenerator
    let mel: NFKMLXVoiceRestoreMel
    let configuration: NFKMLXMossFormer2SRConfiguration

    init(_ c: NFKMLXMossFormer2SRConfiguration) {
        configuration = c
        backbone = NFKMLXMossFormer2SENet(c.backbone)
        generator = NFKMLXMossFormer2SRGenerator(c)
        mel = NFKMLXVoiceRestoreMel(sampleRate: c.sampleRate, nFFT: c.fftSize, hop: c.hopSize, numMels: c.numMels, fMax: c.melMaximumFrequency)
    }

    /// The HiFi-GAN log-mel of a 48 kHz clip, `[1, frames, 80]`.
    func logMel(_ samples: [Float]) -> MLXArray { mel(samples) }

    /// The generator's waveform `[samples]` from a restored mel `[1, frames, 80]`.
    func generated(_ restoredMel: MLXArray) -> [Float] {
        let waveform = generator(restoredMel)
        eval(waveform)
        return waveform.reshaped([-1]).asArray(Float.self)
    }

    /// `decode_one_audio_mossformer2_sr_48k` on a 48 kHz clip: mel → backbone → generator → the
    /// bandwidth substitution.
    func enhance(_ samples: [Float]) -> [Float] {
        let generatedSamples = generated(backbone(logMel(samples)))
        return NFKMossBandwidthSubstitution.substituted(low: samples, high: generatedSamples, configuration: configuration)
    }
}

// MARK: - Backend

private final class NFKMossSRHolder: @unchecked Sendable {
    let net: NFKMLXMossFormer2SRNet
    init(_ net: NFKMLXMossFormer2SRNet) { self.net = net }
}

/// MossFormer2 SR as an InferKit backend: `NFKInputAudio` (any rate; resampled to 48 kHz) → the
/// bandwidth-extended clip at 48 kHz under `NFKOutputAudio`.
@objc(NFKMLXMossFormer2SRBackend)
public final class NFKMLXMossFormer2SRBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKMossSRHolder
    private let identifier: String
    private let outputDirectory: URL

    init(net: NFKMLXMossFormer2SRNet, identifier: String, outputDirectory: URL = FileManager.default.temporaryDirectory) {
        holder = NFKMossSRHolder(net)
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
        let enhanced = holder.net.enhance(matched)
        let url = outputDirectory.appendingPathComponent("mossformer2-sr-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: enhanced, sampleRate: configuration.sampleRate, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(enhanced.count) / Double(configuration.sampleRate),
                                  sampleRate: Double(configuration.sampleRate), channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
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

/// Registration and weight loading for MossFormer2 SR 48K.
@objc(NFKMLXMossFormer2SR_Factory)
public final class NFKMLXMossFormer2SRFactory: NSObject {
    @objc public static let modelName = "mossformer2-sr"
    static let backboneFile = "last_best_checkpoint_m.pt"
    static let generatorFile = "last_best_checkpoint_g.pt"

    static func makeNet(_ configuration: NFKMLXMossFormer2SRConfiguration = .init()) -> NFKMLXMossFormer2SRNet {
        NFKMLXMossFormer2SRNet(configuration)
    }

    /// Builds a backend from the release directory (holding `last_best_checkpoint_m.pt` and
    /// `last_best_checkpoint_g.pt`). A nil directory builds random weights (`isReady` true).
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let directoryURL {
            try loadBackboneWeights(into: net.backbone, from: directoryURL.appendingPathComponent(backboneFile))
            try loadGeneratorWeights(into: net.generator, from: directoryURL.appendingPathComponent(generatorFile))
        }
        return NFKMLXMossFormer2SRBackend(net: net, identifier: modelName)
    }

    /// Registers `mossformer2-sr` with `NFKMLXModelRegistry` (the weights URL is the release directory).
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { directoryURL in try backend(directoryURL: directoryURL) }
    }

    /// The backbone checkpoint is `{"mossformer": {"mossformer.…": …}}`: the container is not one the
    /// native reader unwraps, so its name lands as a second prefix, stripped here before the SE remap.
    static func loadBackboneWeights(into net: NFKMLXMossFormer2SENet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped: [(String, MLXArray)] = checkpoint.arrays.compactMap { key, value in
            let stripped = key.hasPrefix("mossformer.mossformer.") ? String(key.dropFirst("mossformer.".count)) : key
            guard let name = NFKMLXMossFormer2Factory.remapReferenceKey(stripped) else { return nil }
            let tensor: MLXArray
            if value.ndim == 4, checkpoint.needsConvTranspose {
                tensor = value.transposed(0, 2, 3, 1)
            } else if value.ndim == 3, checkpoint.needsConvTranspose {
                tensor = value.transposed(0, 2, 1)
            } else {
                tensor = value
            }
            return (name, tensor)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// The generator checkpoint (`{"generator": …}`) is weight-normed: fuse `g · v / ‖v‖`, then move the
    /// convolutions to channels-last (the transposed `ups` through `[in, out, k]` → `[out, k, in]`) and
    /// the Snake `alpha` from `[1, C, 1]` to `[1, 1, C]`.
    static func loadGeneratorWeights(into net: NFKMLXMossFormer2SRGenerator, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let fused = NFKMLXMusic3.fusedWeightNorm(checkpoint.arrays)
        let mapped: [(String, MLXArray)] = fused.map { key, value in
            guard value.ndim == 3, checkpoint.needsConvTranspose else { return (key, value) }
            return (key, key.hasPrefix("ups.") ? value.transposed(1, 2, 0) : value.transposed(0, 2, 1))
        }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }
}
