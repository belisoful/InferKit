// StoRM (Stochastic Regeneration Model, sp-uhh/storm, MIT): a FEW-STEP follow-on on SGMSE+. A
// DISCRIMINATIVE predictor network produces an initial denoised estimate, then the SGMSE+ score network
// REGENERATES from that estimate — the reverse SDE is re-centered on the predictor's output rather than
// on pure noise, so it needs far fewer steps and the diffusion only has to repair residual artifacts.
//
// Both networks are NCSN++ (NFKMLXNCSNppNet, generalized in NFKMLXSGMSE.swift): the predictor runs in
// discriminative mode (no time embedding, no sigma scaling, 2 input channels → the denoised spectrogram),
// and the score net conditions on [noisy, denoised] (6 input channels for the default `condition="both"`).
// The OUVE scheduler, the FIR resampler, the amplitude compression, and the STFT front end are shared
// with SGMSE+.

import Foundation
import InferKit
import MLX
import MLXNN

/// What the StoRM score network conditions on, beside the diffused state `x`.
@objc public enum NFKMLXStoRMCondition: Int, Sendable {
    /// The noisy observation only (`[y]`).
    case noisy
    /// The denoised estimate only (`[y_denoised]`).
    case postDenoiser
    /// Both, concatenated (`[y, y_denoised]`) — the released default.
    case both
}

/// The StoRM configuration: the shared SGMSE+ geometry (SDE, STFT, NCSN++ base) plus the conditioning
/// mode. The two networks are derived from it — the discriminative predictor and the conditioned score.
public struct NFKMLXStoRMConfiguration: Sendable {
    /// The shared base (SDE θ / σ, STFT, NCSN++ nf / ch_mult / attention / progressive / window). The
    /// `inputChannels` / `conditional` / `scaleBySigma` of the base are overridden per network.
    public var base: NFKMLXSGMSEConfiguration
    /// What the score network conditions on (`both` by default).
    public var condition: NFKMLXStoRMCondition

    public init(base: NFKMLXSGMSEConfiguration = NFKMLXSGMSEConfiguration(), condition: NFKMLXStoRMCondition = .both) {
        self.base = base
        self.condition = condition
    }

    /// The score network's real input-channel count: `x` plus the conditioning (each complex → 2 real).
    var scoreInputChannels: Int { condition == .both ? 6 : 4 }

    /// The discriminative predictor's configuration: reads `y` (2 channels), no time conditioning, no
    /// sigma scaling. Output is the denoised spectrogram.
    var denoiserConfiguration: NFKMLXSGMSEConfiguration {
        var c = base
        c.inputChannels = 2
        c.conditional = false
        c.scaleBySigma = false
        return c
    }

    /// The score network's configuration: reads `x` + the conditioning, with the time embedding and the
    /// sigma scaling (as SGMSE+).
    var scoreConfiguration: NFKMLXSGMSEConfiguration {
        var c = base
        c.inputChannels = scoreInputChannels
        c.conditional = true
        c.scaleBySigma = true
        return c
    }
}

/// The two-network StoRM model. The keys mirror the reference `StochasticRegenerationModel`
/// (`denoiser_net.*`, `score_net.*`), so a converted checkpoint loads with no remap.
public final class NFKMLXStoRMNet: Module {
    @ModuleInfo(key: "denoiser_net") var denoiser: NFKMLXNCSNppNet
    @ModuleInfo(key: "score_net") var score: NFKMLXNCSNppNet

    let config: NFKMLXStoRMConfiguration

    public init(_ config: NFKMLXStoRMConfiguration) {
        self.config = config
        self._denoiser.wrappedValue = NFKMLXNCSNppNet(config.denoiserConfiguration)
        self._score.wrappedValue = NFKMLXNCSNppNet(config.scoreConfiguration)
        super.init()
    }

    /// The discriminative prediction: `y` (packed `[batch, freq, time, 2]`) → the denoised spectrogram
    /// `[batch, freq, time, 2]`. The time argument is a constant (the reference passes `t = ones`).
    public func denoise(_ y: MLXArray) -> MLXArray {
        denoiser(y, sigmas: MLXArray([Float(1)]))
    }
}

// MARK: - Backend + weight loading

private final class NFKStoRMHolder: @unchecked Sendable {
    let net: NFKMLXStoRMNet
    let config: NFKMLXStoRMConfiguration
    init(_ net: NFKMLXStoRMNet, _ config: NFKMLXStoRMConfiguration) {
        self.net = net
        self.config = config
    }
}

/// StoRM speech enhancement / dereverberation as an InferKit backend. Reads `NFKInputAudio` and returns
/// the enhanced clip under `NFKOutputAudio`. The predict-then-regenerate path is multi-step; run it off
/// the render thread.
@objc(NFKMLXStoRMBackend)
public final class NFKMLXStoRMBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKStoRMHolder
    private let identifier: String
    private let seed: UInt64
    private let outputDirectory: URL

    init(net: NFKMLXStoRMNet, config: NFKMLXStoRMConfiguration, identifier: String, seed: UInt64 = 0,
         outputDirectory: URL = FileManager.default.temporaryDirectory) {
        holder = NFKStoRMHolder(net, config)
        self.identifier = identifier
        self.seed = seed
        self.outputDirectory = outputDirectory
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, sampleRate) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
        let input = samples.withUnsafeBufferPointer { MLXArray($0, [1, samples.count]) }
        let enhanced = NFKMLXStoRM.enhance(input, net: holder.net, config: holder.config, seed: seed)
        eval(enhanced)
        let stream = enhanced.reshaped([enhanced.shape[enhanced.ndim - 1]]).asArray(Float.self)
        let length = stream.count

        let url = outputDirectory.appendingPathComponent("storm-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: stream, sampleRate: sampleRate, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(length) / Double(sampleRate),
                                  sampleRate: Double(sampleRate), channelCount: 1)
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

/// Registration, the enhance pipeline, and weight loading for StoRM.
@objc(NFKMLXStoRM)
public final class NFKMLXStoRM: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "storm"

    /// The stochastic-regeneration enhance path (`model.enhance`): peak-normalize, STFT + amplitude-compress
    /// the input, pad the time axis to a multiple of 64, run the discriminative predictor, then regenerate
    /// with the score net's reverse-SDE sampler re-centered on the denoised estimate (no corrector by
    /// default — the point is few steps), undo the compression, invert the STFT, and restore the peak.
    /// `input` is `[1, samples]`.
    public static func enhance(_ input: MLXArray, net: NFKMLXStoRMNet, config: NFKMLXStoRMConfiguration,
                               seed: UInt64 = 0) -> MLXArray {
        let base = config.base
        let originalLength = input.shape[input.ndim - 1]
        let normFactor = maximum(input.abs().max(), MLXArray(Float(1e-8)))
        let normalized = input / normFactor

        let stft = base.stft
        let (re0, im0) = stft.transformComplex(normalized)
        let (yRe0, yIm0) = NFKSGMSESpec.forward(re: re0, im: im0, factor: base.specFactor, exponent: base.specAbsExponent)

        let frames = yRe0.shape[2]
        let padTime = frames % 64 == 0 ? 0 : 64 - frames % 64
        let yRe = padTime > 0 ? MLX.padded(yRe0, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((0, padTime))]) : yRe0
        let yIm = padTime > 0 ? MLX.padded(yIm0, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((0, padTime))]) : yIm0
        let noisy = NFKSGMSESpectrogram(real: yRe, imaginary: yIm)

        // Discriminative prediction: y → y_denoised.
        let denoised4 = net.denoise(stacked([yRe, yIm], axis: -1))          // [1, F, T', 2]
        let denoised = NFKSGMSESpectrogram(real: denoised4[0..., 0..., 0..., 0], imaginary: denoised4[0..., 0..., 0..., 1])

        // Regenerate from the denoised estimate. The SDE centers on y_denoised; the score conditions per
        // the configured mode.
        let conditioning: [NFKSGMSESpectrogram]
        switch config.condition {
        case .noisy: conditioning = [noisy]
        case .postDenoiser: conditioning = [denoised]
        case .both: conditioning = [noisy, denoised]
        }
        let sampler = NFKSGMSESampler(net: net.score, scheduler: NFKMLXOUVEScheduler(base), seed: seed,
                                      observation: denoised, conditioning: conditioning, useCorrector: false)
        let result = sampler.sample()

        let sRe = result.real[0..., 0..., 0 ..< frames]
        let sIm = result.imaginary[0..., 0..., 0 ..< frames]
        let (backRe, backIm) = NFKSGMSESpec.backward(re: sRe, im: sIm, factor: base.specFactor, exponent: base.specAbsExponent)
        let audio = stft.inverseComplex(real: backRe, imaginary: backIm)

        let produced = audio.shape[audio.ndim - 1]
        let cropped = produced >= originalLength ? audio[0..., 0 ..< originalLength] : audio
        return cropped * normFactor
    }

    /// Builds a StoRM backend directly from optional local weights (the EMA safetensors carrying both
    /// networks under `denoiser_net.*` / `score_net.*`). A nil `weightsURL` builds random weights.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        try backend(weightsURL: weightsURL, seed: 0)
    }

    /// Builds a StoRM backend with an explicit sampler seed and configuration.
    public static func backend(weightsURL: URL?, seed: UInt64,
                               config: NFKMLXStoRMConfiguration = NFKMLXStoRMConfiguration()) throws -> any NFKInferenceBackend {
        let net = NFKMLXStoRMNet(config)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXStoRMBackend(net: net, config: config, identifier: modelName, seed: seed)
    }

    /// Downloads the EMA weights from Hugging Face, then builds.
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

    /// Registers StoRM (`storm`) with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads the EMA safetensors — both networks under `denoiser_net.*` / `score_net.*` (the NCSN++ keys
    /// the port mirrors). The only transform is the 4-D Conv2d weight transpose.
    static func loadWeights(into net: NFKMLXStoRMNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let transpose = checkpoint.needsConvTranspose
        let mapped = checkpoint.arrays.map { key, value -> (String, MLXArray) in
            (transpose && value.ndim == 4) ? (key, value.transposed(0, 2, 3, 1)) : (key, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
