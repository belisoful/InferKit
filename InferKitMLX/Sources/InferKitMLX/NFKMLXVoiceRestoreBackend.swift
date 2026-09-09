// VoiceRestore end to end: the BigVGAN mel front end, the conditional-flow-matching midpoint sampler, and
// the audio→audio restoration backend. The mel reproduces BigVGAN's `meldataset.mel_spectrogram`
// (n_fft 1024, hop 256, win 1024, fmin 0, fmax 12000, Hann, center=False with a reflect pad of
// `(n_fft-hop)/2`, magnitude `sqrt(·+1e-9)`, `log(clamp(·, 1e-5))`); the sampler reproduces
// `VoiceRestore.sample` (`torchdiffeq` fixed-step `midpoint`, `times = linspace(0, 1, steps)`, classifier-
// free guidance) with an injectable initial noise for parity.

import Foundation
import InferKit
import MLX
import MLXFFT
import MLXNN

/// The BigVGAN log-mel front end (`meldataset.mel_spectrogram`), reusing the shared Slaney filterbank.
struct NFKMLXVoiceRestoreMel {
    let nFFT: Int, hop: Int, sampleRate: Int, numMels: Int
    let fMax: Float
    let window: [Float]
    let filters: MLXArray                                              // [bins, numMels]

    /// The BigVGAN 24 kHz geometry by default; MossFormer2 SR reads it at 48 kHz over 80 bands to 8 kHz.
    init(sampleRate: Int = 24000, nFFT: Int = 1024, hop: Int = 256, numMels: Int = 100, fMax: Float = 12000) {
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hop = hop
        self.numMels = numMels
        self.fMax = fMax
        let n = nFFT
        window = (0 ..< n).map { 0.5 - 0.5 * cosf(2 * .pi * Float($0) / Float(n)) }   // periodic Hann
        filters = NFKMLXMel.melFilters(sampleRate: sampleRate, bins: n / 2 + 1, nMels: numMels, fMinimum: 0, fMaximum: fMax)
    }

    /// `samples` → `[1, frames, numMels]` log-mel.
    func callAsFunction(_ samples: [Float]) -> MLXArray {
        let pad = (nFFT - hop) / 2                                     // 384, reflect
        var padded: [Float]
        if samples.count > pad {
            padded = (1 ... pad).reversed().map { samples[$0] } + samples
                   + (0 ..< pad).map { samples[samples.count - 2 - $0] }
        } else {
            padded = [Float](repeating: 0, count: pad) + samples + [Float](repeating: 0, count: pad)
        }
        if padded.count < nFFT { padded += [Float](repeating: 0, count: nFFT - padded.count) }
        let frames = 1 + (padded.count - nFFT) / hop                   // center = false
        var frameData = [Float](repeating: 0, count: frames * nFFT)
        for f in 0 ..< frames {
            for j in 0 ..< nFFT { frameData[f * nFFT + j] = padded[f * hop + j] * window[j] }
        }
        let frameArray = frameData.withUnsafeBufferPointer { MLXArray($0, [frames, nFFT]) }
        let spectrum = MLXFFT.rfft(frameArray, axis: 1)                // [frames, bins]
        let magnitude = sqrt(spectrum.realPart() * spectrum.realPart() + spectrum.imaginaryPart() * spectrum.imaginaryPart() + 1e-9)
        let mel = matmul(magnitude, filters)                          // [frames, numMels]
        return log(maximum(mel, MLXArray(1e-5))).reshaped([1, frames, numMels])
    }
}

/// The conditional-flow-matching sampler (`VoiceRestore.sample`): a fixed-step midpoint ODE from `t=0` to
/// `t=1` over `steps` points, `dx/dt = v_θ(x_t, t | degraded_mel)` under classifier-free guidance.
enum NFKMLXVoiceRestoreSampler {
    /// `processed [1, T, 100]` (the degraded mel) → the restored mel `[1, T, 100]`. `noise` is the initial
    /// state `y0` (random when nil); passing the reference's `y0` makes the trajectory reproducible.
    static func sample(_ net: NFKMLXVoiceRestore, processed: MLXArray, steps: Int, cfgStrength: Float,
                       noise: MLXArray? = nil) -> MLXArray {
        let times = (0 ..< steps).map { Float($0) / Float(steps - 1) }
        var y = noise ?? MLXRandom.normal(processed.shape)
        func odeFn(_ t: Float, _ x: MLXArray) -> MLXArray {
            net.guided(x, times: MLXArray([t]), cond: processed, cfgStrength: cfgStrength)
        }
        for i in 0 ..< steps - 1 {
            let dt = times[i + 1] - times[i]
            let k1 = odeFn(times[i], y)
            let k2 = odeFn(times[i] + dt / 2, y + (dt / 2) * k1)       // midpoint
            y = y + dt * k2
            eval(y)
        }
        return y
    }
}

// MARK: - Backend

private final class NFKVRRestoreHolder: @unchecked Sendable {
    let net: NFKMLXVoiceRestore
    let vocoder: NFKMLXBigVGAN
    let mel = NFKMLXVoiceRestoreMel()
    init(net: NFKMLXVoiceRestore, vocoder: NFKMLXBigVGAN) { self.net = net; self.vocoder = vocoder }
}

/// A VoiceRestore speech-restoration backend: `NFKInputAudio` (a degraded clip, resampled to 24 kHz) →
/// the restored `NFKAudioAsset` under `NFKOutputAudio`.
@objc(NFKMLXVoiceRestoreBackend)
public final class NFKMLXVoiceRestoreBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKVRRestoreHolder
    private let identifier: String
    private let outputDirectory: URL
    private let steps: Int
    private let cfgStrength: Float

    init(net: NFKMLXVoiceRestore, vocoder: NFKMLXBigVGAN, identifier: String, steps: Int, cfgStrength: Float,
         outputDirectory: URL = FileManager.default.temporaryDirectory) {
        holder = NFKVRRestoreHolder(net: net, vocoder: vocoder)
        self.identifier = identifier
        self.steps = steps
        self.cfgStrength = cfgStrength
        self.outputDirectory = outputDirectory
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, sampleRate) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
        let input = sampleRate == 24000 ? samples : NFKMLXAudioRate.matched(samples, from: sampleRate, to: 24000)
        let mel = holder.mel(input)
        let restored = NFKMLXVoiceRestoreSampler.sample(holder.net, processed: mel, steps: steps, cfgStrength: cfgStrength)
        let wav = holder.vocoder(restored)                            // [1, T·256, 1]
        eval(wav)
        let stream = wav[0][0..., 0].asArray(Float.self)
        let url = outputDirectory.appendingPathComponent("voicerestore-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: stream, sampleRate: 24000, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(stream.count) / 24000,
                                  sampleRate: 24000, channelCount: 1)
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

// MARK: - Backend factory

extension NFKMLXVoiceRestoreFactory {
    /// Builds a runnable VoiceRestore backend from a transformer checkpoint and a BigVGAN generator.
    @objc(backendWithWeightsURL:vocoderURL:steps:cfgStrength:error:)
    public static func backend(weightsURL: URL, vocoderURL: URL, steps: Int, cfgStrength: Float) throws -> any NFKInferenceBackend {
        let net = makeNet()
        try loadWeights(into: net, from: weightsURL)
        let vocoder = NFKMLXBigVGAN(.init())
        try vocoder.loadWeights(from: vocoderURL)
        return NFKMLXVoiceRestoreBackend(net: net, vocoder: vocoder, identifier: modelName, steps: steps, cfgStrength: cfgStrength)
    }

    /// Builds the backend from a directory holding the transformer checkpoint (`voicerestore.safetensors`
    /// or `pytorch_model.bin`) and `bigvgan_generator.pt`. A nil directory builds random-weight nets.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        let vocoder = NFKMLXBigVGAN(.init())
        if let directoryURL {
            let fm = FileManager.default
            let transformer = ["voicerestore.safetensors", "pytorch_model.bin"]
                .map { directoryURL.appendingPathComponent($0) }
                .first { fm.fileExists(atPath: $0.path) }
            guard let transformer else { throw NFKMLXError.weightsMismatch("voicerestore transformer checkpoint not found in \(directoryURL.path)") }
            try loadWeights(into: net, from: transformer)
            try vocoder.loadWeights(from: directoryURL.appendingPathComponent("bigvgan_generator.pt"))
        }
        return NFKMLXVoiceRestoreBackend(net: net, vocoder: vocoder, identifier: modelName, steps: 32, cfgStrength: 0.5)
    }
}
