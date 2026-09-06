// The DeepFilterNet3 DSP front end, backend, and weight loading. The STFT / ERB / normalization DSP is
// Rust (`libdf`) in the reference; it is reproduced here in MLX + Swift, each step validated numerically
// against a libdf recording (analysis, ERB banding, erb_norm, unit_norm, synthesis all < 4e-6):
//
//   - analysis: left-pad by `nFFT - hop`, frame at `hop`, window with the VORBIS window
//     `sin(π/2·sin²(π(n+0.5)/N))`, rfft, scale `1/N`.
//   - ERB feature: `erb_pow = |spec|² · erb_fb`; `x_dB = 10·log10(erb_pow + 1e-10)`; a per-band EMA
//     mean-normalization `s = x·(1-α) + s·α`, `out = (x - s)/40`, α = 0.99, `s` initialized to
//     `linspace(-60, -90, nbERB)` (`MEAN_NORM_INIT`).
//   - unit_norm (spec feature): the lowest `nbDF` bins, `s = |x|·(1-α) + s·α`, `out = x/√s`, `s`
//     initialized to `linspace(0.001, 0.0001, nbDF)` (`UNIT_NORM_INIT`).
//   - synthesis: `irfft(spec·N)`, window-squared overlap-add at `hop`.

import Foundation
import InferKit
import MLX
import MLXFFT
import MLXNN

/// The DeepFilterNet3 DSP front end, holding the ERB forward bank and the constants the features need.
struct NFKMLXDeepFilterNetDSP {
    let config: NFKMLXDeepFilterNetConfiguration
    let window: MLXArray                                               // [nFFT] Vorbis window
    let erbFB: MLXArray                                                // [bins, nbERB]
    let alpha: Float = 0.99                                            // exp(-hop/sr) rounded, norm_tau = 1

    init(config: NFKMLXDeepFilterNetConfiguration, erbFB: MLXArray) {
        self.config = config
        self.erbFB = erbFB
        let n = config.fftSize
        window = MLXArray((0 ..< n).map { i -> Float in
            let s = sinf(.pi * (Float(i) + 0.5) / Float(n))
            return sinf(.pi / 2 * s * s)
        })
    }

    /// `audio` (length a multiple of `hop`) → `(real, imaginary)` spectra, each `[T, bins]`.
    private func analysis(_ audio: MLXArray) -> (real: MLXArray, imaginary: MLXArray) {
        let n = config.fftSize, hop = config.hopSize
        let leftPad = n - hop
        let padded = MLX.padded(audio, widths: [IntOrPair((leftPad, 0))], mode: .constant)   // [L + leftPad]
        let frames = audio.dim(0) / hop
        var gather = [Int32]()
        gather.reserveCapacity(frames * n)
        for t in 0 ..< frames {
            for k in 0 ..< n { gather.append(Int32(t * hop + k)) }
        }
        let framed = take(padded, MLXArray(gather), axis: 0).reshaped([frames, n]) * window.reshaped([1, n])
        let spectrum = MLXFFT.rfft(framed, axis: 1) / Float(n)          // [T, bins] complex
        return (spectrum.realPart(), spectrum.imaginaryPart())
    }

    /// `audio` → `(spec [1, T, bins, 2], featERB [1, T, nbERB, 1], featSpec [1, T, nbDF, 2])`.
    func features(_ audioSamples: [Float]) -> (spec: MLXArray, featERB: MLXArray, featSpec: MLXArray) {
        let audio = audioSamples.withUnsafeBufferPointer { MLXArray($0, [audioSamples.count]) }
        let (real, imaginary) = analysis(audio)                        // [T, bins]
        let t = real.dim(0), bins = config.bins, nbERB = config.nbERB, nbDF = config.nbDF

        // ERB power banding and the dB + EMA mean-normalization (per-band scan in Swift).
        let power = (real * real + imaginary * imaginary)              // [T, bins]
        let erbPow = matmul(power, erbFB)                              // [T, nbERB]
        eval(erbPow)
        let pow = erbPow.asArray(Float.self)
        var stateERB = (0 ..< nbERB).map { -60 + (-30) * Float($0) / Float(nbERB - 1) }   // linspace(-60,-90)
        var featERB = [Float](repeating: 0, count: t * nbERB)
        for frame in 0 ..< t {
            for band in 0 ..< nbERB {
                let xdb = 10 * log10f(pow[frame * nbERB + band] + 1e-10)
                stateERB[band] = xdb * (1 - alpha) + stateERB[band] * alpha
                featERB[frame * nbERB + band] = (xdb - stateERB[band]) / 40
            }
        }

        // unit_norm on the lowest nbDF bins (EMA magnitude, per-bin scan in Swift).
        let lowReal = real[0..., .stride(to: nbDF)], lowImag = imaginary[0..., .stride(to: nbDF)]
        eval(lowReal, lowImag)
        let lr = lowReal.asArray(Float.self), li = lowImag.asArray(Float.self)
        var stateDF = (0 ..< nbDF).map { 0.001 + (-0.0009) * Float($0) / Float(nbDF - 1) }   // linspace(0.001,0.0001)
        var featSpec = [Float](repeating: 0, count: t * nbDF * 2)
        for frame in 0 ..< t {
            for bin in 0 ..< nbDF {
                let re = lr[frame * nbDF + bin], im = li[frame * nbDF + bin]
                stateDF[bin] = sqrtf(re * re + im * im) * (1 - alpha) + stateDF[bin] * alpha
                let inv = 1 / sqrtf(stateDF[bin])
                featSpec[(frame * nbDF + bin) * 2] = re * inv
                featSpec[(frame * nbDF + bin) * 2 + 1] = im * inv
            }
        }

        let spec = stacked([real, imaginary], axis: 2).reshaped([1, t, bins, 2])
        let featERBArray = MLXArray(featERB).reshaped([1, t, nbERB, 1])
        let featSpecArray = MLXArray(featSpec).reshaped([1, t, nbDF, 2])
        return (spec, featERBArray, featSpecArray)
    }

    /// Enhanced spectrum `[1, T, bins, 2]` → time samples (length `T·hop`), window-squared overlap-add.
    func synthesis(_ spec: MLXArray) -> [Float] {
        let n = config.fftSize, hop = config.hopSize
        let t = spec.dim(1)
        let real = spec[0, 0..., 0..., 0], imaginary = spec[0, 0..., 0..., 1]   // [T, bins]
        let complex = real.asType(.complex64) + imaginary.asType(.complex64) * MLXArray(real: 0, imaginary: 1)
        let framed = MLXFFT.irfft(complex * Float(n), n: n, axis: 1)   // [T, n] real (undo the analysis /N)
        eval(framed)
        let frameValues = framed.asArray(Float.self)
        let windowValues = window.asArray(Float.self)
        let outLength = t * hop
        var output = [Float](repeating: 0, count: outLength)
        var norm = [Float](repeating: 0, count: outLength)
        for frame in 0 ..< t {
            let start = frame * hop
            for k in 0 ..< n where start + k < outLength {
                output[start + k] += frameValues[frame * n + k] * windowValues[k]
                norm[start + k] += windowValues[k] * windowValues[k]
            }
        }
        for i in 0 ..< outLength where norm[i] > 1e-10 { output[i] /= norm[i] }
        return output
    }
}

// MARK: - Backend

final class NFKDFHolder: @unchecked Sendable {
    let net: NFKMLXDeepFilterNet
    let dsp: NFKMLXDeepFilterNetDSP
    init(_ net: NFKMLXDeepFilterNet) {
        self.net = net
        dsp = NFKMLXDeepFilterNetDSP(config: net.config, erbFB: net.erbFB)
    }
}

/// A DeepFilterNet3 speech-denoising backend: `NFKInputAudio` (resampled to 48 kHz) → an enhanced
/// `NFKAudioAsset` under `NFKOutputAudio`.
@objc(NFKMLXDeepFilterNetBackend)
public final class NFKMLXDeepFilterNetBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKDFHolder
    private let identifier: String
    private let outputDirectory: URL

    init(net: NFKMLXDeepFilterNet, identifier: String,
         outputDirectory: URL = FileManager.default.temporaryDirectory) {
        holder = NFKDFHolder(net)
        self.identifier = identifier
        self.outputDirectory = outputDirectory
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, sampleRate) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
        let config = holder.net.config
        let input = sampleRate == config.sampleRate ? samples
            : NFKMLXAudioRate.matched(samples, from: sampleRate, to: config.sampleRate)
        let enhanced = Self.enhance(input, net: holder.net, dsp: holder.dsp)
        let url = outputDirectory.appendingPathComponent("deepfilternet-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: enhanced, sampleRate: config.sampleRate, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(enhanced.count) / Double(config.sampleRate),
                                  sampleRate: Double(config.sampleRate), channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
    }

    /// The full `enhance()`: right-pad by `nFFT` to compensate the STFT delay, run analysis → the net →
    /// synthesis, then trim `[nFFT-hop : origLen + nFFT-hop]` to realign and restore the length.
    static func enhance(_ samples: [Float], net: NFKMLXDeepFilterNet,
                        dsp: NFKMLXDeepFilterNetDSP? = nil) -> [Float] {
        let config = net.config
        let dsp = dsp ?? NFKMLXDeepFilterNetDSP(config: config, erbFB: net.erbFB)
        let hop = config.hopSize, n = config.fftSize
        let origLen = samples.count
        var padded = samples + [Float](repeating: 0, count: n)
        let remainder = padded.count % hop
        if remainder != 0 { padded += [Float](repeating: 0, count: hop - remainder) }
        let (spec, featERB, featSpec) = dsp.features(padded)
        let (specE, _, _, _) = net(spec: spec, featERB: featERB, featSpec: featSpec)
        eval(specE)
        let audioOut = dsp.synthesis(specE)
        let d = n - hop
        let end = min(origLen + d, audioOut.count)
        guard d < end else { return audioOut }
        return Array(audioOut[d ..< end])
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

// MARK: - Registration and weight loading

/// Registration and weight loading for DeepFilterNet3.
@objc(NFKMLXDeepFilterNet_Factory)
public final class NFKMLXDeepFilterNetFactory: NSObject {
    @objc public static let modelName = "deepfilternet3"

    static func makeNet(_ config: NFKMLXDeepFilterNetConfiguration = .init()) -> NFKMLXDeepFilterNet {
        let net = NFKMLXDeepFilterNet(config)
        net.train(false)                                              // BatchNorm reads its running statistics
        return net
    }

    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL { try loadWeights(into: net, from: weightsURL) }
        return NFKMLXDeepFilterNetBackend(net: net, identifier: modelName)
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

    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a released DeepFilterNet3 checkpoint into `DfNet`: fold the GRUs into indexed cells, remap the
    /// `Conv2dNormAct` / `SqueezedGRU_S` Sequential names, and transpose the convolutions (the ERB
    /// decoder's `convt2`/`convt1` are grouped `ConvTranspose2d`).
    static func loadWeights(into net: NFKMLXDeepFilterNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let folded = foldGRUs(checkpoint.arrays)
        let mapped: [(String, MLXArray)] = folded.compactMap { key, value in
            guard let name = remapReferenceKey(key) else { return nil }
            let tensor: MLXArray
            if value.ndim == 4, checkpoint.needsConvTranspose {
                if name.contains(".convt2.conv.") || name.contains(".convt1.conv.") {
                    // Depthwise ConvTranspose2d: PyTorch [in, out/g, kH, kW] → MLX [out, kH, kW, in/g].
                    tensor = deconvWeight(value, groups: value.dim(0))
                } else {
                    tensor = value.transposed(0, 2, 3, 1)
                }
            } else {
                tensor = value
            }
            return (name, tensor)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// A grouped `ConvTranspose2d` weight `[in, out/groups, kH, kW]` → MLX `[out, kH, kW, in/groups]`
    /// (shared with GTCRN's grouped-transpose handling).
    static func deconvWeight(_ weight: MLXArray, groups: Int) -> MLXArray {
        let inC = weight.dim(0), outPerGroup = weight.dim(1), kH = weight.dim(2), kW = weight.dim(3)
        let inPerGroup = inC / groups, out = outPerGroup * groups
        return weight.reshaped([groups, inPerGroup, outPerGroup, kH, kW])
            .transposed(0, 2, 3, 4, 1)
            .reshaped([out, kH, kW, inPerGroup])
    }

    private static let paddedConvs: Set<String> = ["erb_conv0", "df_conv0", "df_convp"]
    private static let separableConvs: Set<String> = ["erb_conv1", "erb_conv2", "erb_conv3", "df_conv0",
                                                       "df_conv1", "convt3", "convt2", "convt1", "df_convp"]
    private static let convModules: Set<String> = ["erb_conv0", "erb_conv1", "erb_conv2", "erb_conv3",
                                                   "df_conv0", "df_conv1", "conv3p", "conv2p", "conv1p",
                                                   "conv0p", "convt3", "convt2", "convt1", "conv0_out", "df_convp"]
    private static let sequentialWrappers: Set<String> = ["linear_in", "linear_out", "df_fc_emb",
                                                          "lsnr_fc", "df_out", "df_fc_a"]

    /// Maps a reference key onto the module names. `num_batches_tracked` counters are dropped; `mask.` is
    /// the top-level `erb_inv_fb`; a `Conv2dNormAct` Sequential index (0/1/2/…) becomes `conv` / `conv_pw`
    /// / `norm` by the layout rule (a causal time-pad shifts the indices by one; a pointwise 1x1 sits
    /// between the conv and the norm); a `Sequential(module, act)` wrapper drops its `.0`.
    static func remapReferenceKey(_ key: String) -> String? {
        if key.hasSuffix("num_batches_tracked") { return nil }
        if key == "mask.erb_inv_fb" { return "erb_inv_fb" }
        var comps = key.split(separator: ".").map(String.init)
        for i in 0 ..< comps.count - 1 {
            let name = comps[i]
            guard let idx = Int(comps[i + 1]) else { continue }
            if convModules.contains(name) {
                let offset = paddedConvs.contains(name) ? 1 : 0
                if idx == offset {
                    comps[i + 1] = "conv"
                } else if separableConvs.contains(name), idx == offset + 1 {
                    comps[i + 1] = "conv_pw"
                } else {
                    comps[i + 1] = "norm"
                }
                return comps.joined(separator: ".")
            }
            if sequentialWrappers.contains(name), idx == 0 {
                comps.remove(at: i + 1)                               // drop the Sequential index
                return comps.joined(separator: ".")
            }
        }
        return key
    }

    /// Folds every `SqueezedGRU_S` `nn.GRU` (single- or multi-layer) into indexed `NFKMLXGRUCell`
    /// parameters: `<base>gru.weight_ih_l{i}` → `<base>gru.{i}.Wx/Wh/b/bhn`, combining the two PyTorch
    /// biases the way MLXNN's GRU expects.
    static func foldGRUs(_ arrays: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        var handled = Set<String>()
        for key in arrays.keys where key.contains(".gru.weight_ih_l") {
            guard let range = key.range(of: ".weight_ih_l") else { continue }
            let base = String(key[key.startIndex ..< range.lowerBound])   // "...gru"
            let layer = String(key[range.upperBound...])                   // "0", "1", …
            guard let weightIH = arrays["\(base).weight_ih_l\(layer)"],
                  let weightHH = arrays["\(base).weight_hh_l\(layer)"],
                  let biasIH = arrays["\(base).bias_ih_l\(layer)"],
                  let biasHH = arrays["\(base).bias_hh_l\(layer)"] else { continue }
            let hidden = weightIH.dim(0) / 3
            let hiddenRZ = MLX.padded(biasHH[0 ..< 2 * hidden], widths: [IntOrPair((0, hidden))], mode: .constant)
            out["\(base).\(layer).Wx"] = weightIH
            out["\(base).\(layer).Wh"] = weightHH
            out["\(base).\(layer).b"] = biasIH + hiddenRZ
            out["\(base).\(layer).bhn"] = biasHH[2 * hidden ..< 3 * hidden]
            for suffix in ["weight_ih_l", "weight_hh_l", "bias_ih_l", "bias_hh_l"] {
                handled.insert("\(base).\(suffix)\(layer)")
            }
        }
        for (key, value) in arrays where !handled.contains(key) { out[key] = value }
        return out
    }
}
