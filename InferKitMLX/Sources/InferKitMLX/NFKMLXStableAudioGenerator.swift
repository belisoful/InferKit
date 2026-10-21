//
//  NFKMLXStableAudioGenerator.swift
//  InferKitMLX
//
//  Stable Audio Open 1.0 text-to-audio from a downloaded diffusers release directory, its two stages
//  held as an `NFKMLXResidency` says: the text stage (t5-base and the seconds conditioners) and the
//  pipeline stage (the diffusion transformer and the Oobleck autoencoder).
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXRandom

/// The conditioning one prompt gives the transformer: the text-and-seconds sequence and the global token.
struct NFKStableAudioConditioning {
    /// `[1, L + 2, conditioningDim]`: the masked text states, then the start and end tokens.
    let context: MLXArray
    /// `[1, 1, 2 · conditioningDim]`: the start and end tokens side by side.
    let global: MLXArray
}

/// The text stage: t5-base, the projection model, and the tokenizer feeding them.
final class NFKStableAudioTextStage {
    let encoder: NFKMLXT5EncoderNet
    let projection: NFKMLXStableAudioProjectionNet

    init(encoder: NFKMLXT5EncoderNet, projection: NFKMLXStableAudioProjectionNet) {
        self.encoder = encoder
        self.projection = projection
    }

    /// The text states for `prompt`, projected and masked: the pipeline multiplies every padded position
    /// by zero after the projection.
    func text(_ prompt: NFKMLXT5Prompt) -> MLXArray {
        let mask = prompt.maskArray
        let states = projection.projectedText(encoder(prompt.tokens, mask: mask))
        return states * mask.expandedDimensions(axis: -1).asType(states.dtype)
    }

    /// The conditioning for `prompt` over `start ..< end` seconds, as `encode_prompt` and
    /// `encode_duration` assemble it.
    func conditioning(_ prompt: NFKMLXT5Prompt, start: Float, end: Float) -> NFKStableAudioConditioning {
        let seconds = projection.seconds(start: MLXArray([start]), end: MLXArray([end]))
        return NFKStableAudioConditioning(
            context: concatenated([text(prompt), seconds.start, seconds.end], axis: 1),
            global: concatenated([seconds.start, seconds.end], axis: -1))
    }
}

/// Stable Audio Open's sampling loop and decode over loaded networks.
///
/// @discussion ``latents(context:global:negativeContext:steps:guidance:initial:encoded:noise:progress:)`` follows
/// diffusers' `StableAudioPipeline`: the initial latents scaled by the sampler's `init_noise_sigma`, each
/// step the transformer over the classifier-free guidance pair `[unconditional, conditional]` at the
/// preconditioned input, the guided v-prediction `u + g·(c − u)`, and a cosine DPM-Solver++ SDE step.
/// The unconditional branch is the negative prompt's conditioning, or zeros where none is given.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXStableAudioPipeline {
    public let transformer: NFKMLXStableAudioDiTNet
    public let autoencoder: NFKMLXOobleckAutoencoderNet
    public let schedulerConfiguration: NFKMLXCosineDPMSolverScheduler.Configuration

    public init(transformer: NFKMLXStableAudioDiTNet, autoencoder: NFKMLXOobleckAutoencoderNet,
                schedulerConfiguration: NFKMLXCosineDPMSolverScheduler.Configuration = .init()) {
        self.transformer = transformer
        self.autoencoder = autoencoder
        self.schedulerConfiguration = schedulerConfiguration
    }

    /// The latent window, `[1, sampleSize, inChannels]`.
    public var latentShape: [Int] { [1, transformer.configuration.sampleSize, transformer.configuration.inChannels] }

    /// The longest clip the window holds, in seconds (47.55 for the release).
    public var maximumDuration: Double {
        Double(transformer.configuration.sampleSize * autoencoder.configuration.hopLength)
            / Double(autoencoder.configuration.samplingRate)
    }

    /// The scale `prepare_latents` gives standard-normal latents, `√(σ_max² + 1)`.
    var initialNoiseScale: Float {
        Float((Double(schedulerConfiguration.sigmaMax) * Double(schedulerConfiguration.sigmaMax) + 1).squareRoot())
    }

    /// Denoises from `initial` (standard-normal latents of ``latentShape``, plus an encoded waveform's
    /// latents where a clip starts from one) and returns the final latents.
    ///
    /// - Parameters:
    ///   - context: the conditional `[1, L, d]` sequence.
    ///   - global: the `[1, 1, 2d]` global token, shared by both branches of the guidance pair.
    ///   - negativeContext: the unconditional sequence; zeros where nil.
    ///   - steps: the sampler's step count.
    ///   - guidance: the classifier-free guidance scale. Above 1 the transformer runs the guidance pair;
    ///     at 1 or below it runs the conditional branch alone.
    ///   - initial: standard-normal latents of ``latentShape``, scaled by the sampler's initial sigma.
    ///   - noise: the sampler's standard normal for step `index`.
    ///   - encoded: an encoded waveform's latents, added to the scaled noise.
    ///   - progress: called after each step with its index and the latents it produced; returning false
    ///     stops the loop with nil.
    public func latents(context: MLXArray, global: MLXArray, negativeContext: MLXArray?, steps: Int, guidance: Float,
                        initial: MLXArray, encoded: MLXArray? = nil, noise: (Int) -> MLXArray,
                        progress: (Int, MLXArray) -> Bool = { _, _ in true }) -> MLXArray? {
        let guides = guidance > 1
        let pairedContext = guides
            ? concatenated([negativeContext ?? MLXArray.zeros(like: context), context], axis: 0) : context
        let pairedGlobal = guides ? concatenated([global, global], axis: 0) : global
        let scheduler = NFKMLXCosineDPMSolverScheduler(steps: steps, configuration: schedulerConfiguration)
        var sample = initial * initialNoiseScale
        if let encoded {
            sample = encoded + sample
        }
        for index in 0 ..< steps {
            let input = guides ? concatenated([sample, sample], axis: 0) : sample
            let prediction = transformer(input * scheduler.inputScale, timestep: MLXArray([scheduler.timestep]),
                                         context: pairedContext, global: pairedGlobal)
            var guided = prediction
            if guides {
                let unconditional = prediction[0 ..< 1], conditional = prediction[1 ..< 2]
                guided = unconditional + guidance * (conditional - unconditional)
            }
            sample = scheduler.step(modelOutput: guided, sample: sample, noise: noise(index))
            eval(sample)
            guard progress(index, sample) else { return nil }
        }
        return sample
    }

    /// The waveform `[samples, audioChannels]` the latents decode to over `start ..< end` seconds, cropped
    /// as the pipeline crops it (`int(seconds · sampling_rate)`). A span past the decoded window is empty, as
    /// the reference's slice is.
    public func waveform(_ latents: MLXArray, start: Double, end: Double) -> MLXArray {
        let decoded = autoencoder.decode(latents)[0]
        let rate = Double(autoencoder.configuration.samplingRate)
        let last = min(max(Int(end * rate), 0), decoded.dim(0))
        let first = min(max(Int(start * rate), 0), last)
        return decoded[first ..< last]
    }

    /// The latents an initial waveform `[samples, channels]` contributes: zero-padded or cropped to the
    /// window, a mono clip repeated to stereo or a stereo clip averaged to mono, encoded, and sampled from
    /// the posterior with `noise`, as `prepare_latents` does.
    ///
    /// - Throws: `NFKMLXError.unsupportedInput` for a waveform that is not `[samples, channels]` with one or
    ///   two channels, which the reference refuses too.
    public func encodedLatents(_ waveform: MLXArray, noise: MLXArray) throws -> MLXArray {
        let channels = autoencoder.configuration.audioChannels
        let windowSamples = transformer.configuration.sampleSize * autoencoder.configuration.hopLength
        guard waveform.ndim == 2, (1 ... 2).contains(waveform.dim(1)) else { throw NFKMLXError.unsupportedInput }
        var audio = waveform
        if audio.dim(1) == 1 && channels == 2 {
            audio = concatenated([audio, audio], axis: 1)
        } else if audio.dim(1) == 2 && channels == 1 {
            audio = audio.mean(axis: 1, keepDims: true)
        }
        guard audio.dim(1) == channels else { throw NFKMLXError.unsupportedInput }
        audio = audio[0 ..< min(audio.dim(0), windowSamples)]
        if audio.dim(0) < windowSamples {
            audio = padded(audio, widths: [.init((0, windowSamples - audio.dim(0))), .init((0, 0))])
        }
        return autoencoder.sample(audio.expandedDimensions(axis: 0), noise: noise)
    }
}

/// The request-parameter keys the Stable Audio backend reads beyond the core's, so an Objective-C caller
/// sets them the way it sets `NFKParameterDurationSeconds`.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXStableAudioParameterKey)
public final class NFKMLXStableAudioParameterKey: NSObject {
    /// Where the clip starts within the model's window, in seconds (`NSNumber`, default 0): the start
    /// seconds conditioner reads it, and the waveform is cropped from it. `NFKParameterDurationSeconds`
    /// runs from here. Introduced in InferKit 0.4.0.
    @objc public static let startSeconds = "NFKMLXParameterStableAudioStartSeconds"
}

/// Stable Audio Open 1.0 text-to-audio, assembled from a diffusers release directory.
///
/// @discussion The release layout is `text_encoder/` (t5-base), `tokenizer/` (its SentencePiece model),
/// `projection_model/`, `transformer/`, `vae/`, and `scheduler/`. ``audio(forPrompt:negativePrompt:startSeconds:endSeconds:initialAudio:seed:progress:)``
/// encodes the prompt and its seconds, denoises the release's 1024-frame latent window (47.55 seconds at
/// 44.1 kHz) with classifier-free guidance, decodes it, and crops the requested span. Every stage runs at
/// float32, the release's precision. The glue is measured against diffusers' `StableAudioPipeline`. The
/// weights are under the Stability AI Community License.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXStableAudioGenerator)
public final class NFKMLXStableAudioGenerator: NSObject {

    /// A name for the model the factories produce.
    @objc public static let modelName = "stable-audio-open-1.0"

    /// The Hugging Face repository the download factories read when none is named. The repository is
    /// gated: its license is accepted once on the Hub, and the token is `NFKHFHub.defaultAccessToken`.
    @objc public static let releaseRepo = "stabilityai/stable-audio-open-1.0"

    /// The denoising steps, 100 by the reference's default.
    @objc public var steps: Int = 100
    /// The classifier-free guidance scale, 7 by the reference's default. Above 1 the clip guides against
    /// the negative prompt, or against zeroed conditioning where none is given.
    @objc public var guidance: Float = 7

    /// The output sample rate, 44100 for the release.
    @objc public var sampleRate: Int { autoencoderConfiguration.samplingRate }
    /// The output channels, 2 for the release.
    @objc public var channelCount: Int { autoencoderConfiguration.audioChannels }
    /// The longest clip, `endSeconds − startSeconds`, the latent window holds (47.55 seconds for the release).
    @objc public var maximumDuration: Double {
        Double(transformerConfiguration.sampleSize * autoencoderConfiguration.hopLength)
            / Double(autoencoderConfiguration.samplingRate)
    }

    /// Whether the text stage and the pipeline stay loaded together between clips.
    @objc public var holdsStagesResident: Bool { staging.resident }

    private let staging: NFKMLXStagedModel
    private let textStage: NFKMLXStage<NFKStableAudioTextStage>
    private let pipelineStage: NFKMLXStage<NFKMLXStableAudioPipeline>
    private let segmenter: NFKMLXSentencePieceSegmenter
    private let textLength: Int
    private let projectionConfiguration: NFKMLXStableAudioProjectionConfiguration
    private let transformerConfiguration: NFKMLXStableAudioDiTConfiguration
    private let autoencoderConfiguration: NFKMLXOobleckConfiguration

    init(resident: Bool, segmenter: NFKMLXSentencePieceSegmenter, textLength: Int,
         projection: NFKMLXStableAudioProjectionConfiguration, transformer: NFKMLXStableAudioDiTConfiguration,
         autoencoder: NFKMLXOobleckConfiguration, loadText: @escaping () throws -> NFKStableAudioTextStage,
         loadPipeline: @escaping () throws -> NFKMLXStableAudioPipeline) {
        staging = NFKMLXStagedModel(resident: resident)
        textStage = NFKMLXStage(loadText)
        pipelineStage = NFKMLXStage(loadPipeline)
        self.segmenter = segmenter
        self.textLength = textLength
        projectionConfiguration = projection
        transformerConfiguration = transformer
        autoencoderConfiguration = autoencoder
        super.init()
    }

    /// Loads both stages now, for a resident release.
    func loadResident() throws {
        try staging.use(textStage) { _ in }
        try staging.use(pipelineStage) { _ in }
    }

    /// `text` tokenized as the release tokenizer pads it: the end token, then padding to `textLength`.
    func prompt(_ text: String) -> NFKMLXT5Prompt {
        NFKMLXT5Prompt(text, segmenter: segmenter, length: textLength)
    }

    // MARK: Factories

    /// Assembles the model from a downloaded release directory, holding it as
    /// ``NFKMLXResidency/automatic`` decides.
    @objc(generatorWithDirectoryURL:error:)
    public static func generator(directoryURL: URL) throws -> NFKMLXStableAudioGenerator {
        try generator(directoryURL: directoryURL, residency: .automatic)
    }

    /// Assembles the model from a downloaded release directory, holding it as `residency` says.
    ///
    /// @discussion Every stage loads at float32. The release is about 5.3 GB, so a machine that holds it
    /// keeps both stages resident under ``NFKMLXResidency/automatic``. The release has no routed experts,
    /// so ``NFKMLXResidency/paged`` holds it as ``NFKMLXResidency/staged`` does.
    @objc(generatorWithDirectoryURL:residency:error:)
    public static func generator(directoryURL: URL, residency: NFKMLXResidency) throws -> NFKMLXStableAudioGenerator {
        let directory = { (name: String) in directoryURL.appendingPathComponent(name) }
        let t5 = try NFKMLXT5Encoder.configuration(fromHuggingFace: directory("text_encoder/config.json"))
        let segmenter = NFKMLXSentencePieceSegmenter(model: try NFKMLXSentencePieceModel(
            contentsOf: directory("tokenizer/spiece.model")))
        let textLength = try tokenizerLength(directory("tokenizer/tokenizer_config.json"))
        let projection = try NFKMLXStableAudioProjectionConfiguration.configuration(
            fromHuggingFace: directory("projection_model/config.json"))
        let transformer = try NFKMLXStableAudioDiTConfiguration.configuration(
            fromHuggingFace: directory("transformer/config.json"))
        let autoencoder = try NFKMLXOobleckConfiguration.configuration(fromHuggingFace: directory("vae/config.json"))
        let scheduler = try NFKMLXCosineDPMSolverScheduler.Configuration.configuration(
            fromHuggingFace: directory("scheduler/scheduler_config.json"))
        let projectionURL = directory("projection_model/diffusion_pytorch_model.safetensors")
        let vaeURL = directory("vae/diffusion_pytorch_model.safetensors")

        let plan = try NFKMLXResidencyBudget.plan(
            [NFKMLXStageFootprint(bytes: try NFKMLXStageWeights.bytes(inDirectory: directory("text_encoder"), holding: .float32)
                                     + NFKMLXStageWeights.bytes(inDirectory: directory("projection_model"), holding: .float32)),
             NFKMLXStageFootprint(bytes: try NFKMLXStageWeights.bytes(inDirectory: directory("transformer"), holding: .float32)
                                     + NFKMLXStageWeights.bytes(inDirectory: directory("vae"), holding: .float32))],
            residency: residency, budget: NFKMLXResidencyBudget.current())
        let generator = NFKMLXStableAudioGenerator(
            resident: plan.holdsStagesResident, segmenter: segmenter, textLength: textLength, projection: projection,
            transformer: transformer, autoencoder: autoencoder,
            loadText: {
                let encoder = NFKMLXT5Encoder.makeNet(t5)
                try NFKMLXT5Encoder.loadWeights(into: encoder, from: directory("text_encoder"), dtype: .float32)
                let net = NFKMLXStableAudioProjectionNet(projection)
                try NFKMLXStableAudio.loadProjectionWeights(into: net, from: projectionURL)
                return NFKStableAudioTextStage(encoder: encoder, projection: net)
            },
            loadPipeline: {
                let net = NFKMLXStableAudioDiTNet(transformer)
                try NFKMLXStableAudio.loadTransformerWeights(into: net, fromDirectory: directory("transformer"))
                let vae = NFKMLXOobleckAutoencoderNet(autoencoder)
                try NFKMLXStableAudio.loadAutoencoderWeights(into: vae, from: vaeURL)
                return NFKMLXStableAudioPipeline(transformer: net, autoencoder: vae, schedulerConfiguration: scheduler)
            })
        if plan.holdsStagesResident {
            try generator.loadResident()
        }
        return generator
    }

    /// The release tokenizer's `model_max_length`, the length every prompt pads to.
    static func tokenizerLength(_ url: URL) throws -> Int {
        let json = try NFKMLXStableAudio.json(url)
        guard let length = (json["model_max_length"] as? NSNumber)?.intValue, length > 1, length <= 4096 else {
            throw NFKMLXError.unsupportedConfiguration("the tokenizer states no usable model_max_length")
        }
        return length
    }

    /// The component folders a download fetches.
    static let releaseComponents = [
        NFKMLXReleaseComponent(required: ["model_index.json", "scheduler/scheduler_config.json"]),
        NFKMLXReleaseComponent(required: ["text_encoder/config.json", "text_encoder/model.safetensors"]),
        NFKMLXReleaseComponent(required: ["tokenizer/spiece.model", "tokenizer/tokenizer_config.json"],
                               optional: ["tokenizer/special_tokens_map.json", "tokenizer/tokenizer.json"]),
        NFKMLXReleaseComponent(required: ["projection_model/config.json",
                                          "projection_model/diffusion_pytorch_model.safetensors"]),
        NFKMLXReleaseComponent(required: ["transformer/config.json"],
                               weights: ["transformer/diffusion_pytorch_model.safetensors",
                                         "transformer/diffusion_pytorch_model.safetensors.index.json"]),
        NFKMLXReleaseComponent(required: ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]),
        NFKMLXReleaseComponent(required: [], optional: ["LICENSE.md"]),
    ]

    /// Downloads the release (``releaseRepo`` when `repo` is nil) and assembles the model, holding it
    /// as `residency` says.
    ///
    /// @discussion The release is about 5.3 GB. A file already in the cache is not fetched again. The call
    /// blocks on the network, so run it off the main and render threads. The weights are under the
    /// Stability AI Community License.
    @objc(generatorWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func generator(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency) throws -> NFKMLXStableAudioGenerator {
        try NFKMLXBuildHolds.keeping {
            let directory = try NFKMLXReleaseDownload.directory(repo: repo ?? releaseRepo, revision: revision,
                                                                cacheDirectoryURL: cacheDirectoryURL,
                                                                components: releaseComponents)
            return try generator(directoryURL: directory, residency: residency)
        }
    }

    /// The asynchronous form of ``generator(repo:revision:cacheDirectoryURL:residency:)``. The handler
    /// runs on a background queue.
    @objc(generatorWithRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func generator(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency,
                                 completionHandler: @escaping (NFKMLXStableAudioGenerator?, Error?) -> Void) {
        NFKMLXReleaseDownload.asyncValue(completionHandler) {
            try generator(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL, residency: residency)
        }
    }

    // MARK: Generation

    /// The most sampling steps a clip takes, the sampler's 1000 training timesteps.
    @objc public static let maximumSteps = 1000

    /// Checks a span against the conditioners' range and the latent window, as `check_inputs` does, and the
    /// step count. The pipeline crops the decoded window at the span's own seconds, so a span that ends
    /// past the window is refused: the reference would return an empty clip.
    static func validate(start: Double, end: Double, steps: Int, conditionerRange: ClosedRange<Double>,
                         window: Double) throws {
        guard end >= start, conditionerRange.contains(start), conditionerRange.contains(end) else {
            throw NFKMLXError.unsupportedConfiguration(
                "a clip runs from start to end seconds within \(conditionerRange.lowerBound)...\(conditionerRange.upperBound)")
        }
        guard end <= window else {
            throw NFKMLXError.unsupportedConfiguration("a clip ends within the model's \(window)-second window")
        }
        guard (1 ... maximumSteps).contains(steps) else {
            throw NFKMLXError.unsupportedConfiguration("a clip takes 1 to \(maximumSteps) steps")
        }
    }

    func validate(start: Double, end: Double) throws {
        try Self.validate(start: start, end: end, steps: steps,
                          conditionerRange: Double(projectionConfiguration.minValue) ... Double(projectionConfiguration.maxValue),
                          window: maximumDuration)
    }

    /// Generates a clip for `prompt`, `[samples, channelCount]` at ``sampleRate`` in `-1...1` (unclipped,
    /// as the reference returns it), covering `startSeconds ..< endSeconds`.
    ///
    /// @discussion `endSeconds` nil is ``maximumDuration``, the reference's default. Above a ``guidance``
    /// of 1 the clip guides against `negativePrompt`, or against zeroed conditioning where none is given.
    /// `initialAudio` `[samples, channels]` at ``sampleRate``, with one or two channels, adds that
    /// waveform's latents to the initial noise. A span that ends past ``maximumDuration``, or a step count
    /// outside `1 ... maximumSteps`, throws.
    /// `progress` receives the fraction done and returns false to stop, which returns nil.
    public func audio(forPrompt prompt: String, negativePrompt: String? = nil, startSeconds: Double = 0,
                      endSeconds: Double? = nil, initialAudio: MLXArray? = nil, seed: UInt64 = 0,
                      progress: (Double) -> Bool = { _ in true }) throws -> MLXArray? {
        let end = endSeconds ?? maximumDuration
        try validate(start: startSeconds, end: end)
        let guides = guidance > 1
        let tokens = self.prompt(prompt)
        let negativeTokens = guides ? negativePrompt.map { self.prompt($0) } : nil
        return try staging.exclusively {
            let conditioning = try staging.with(textStage) { stage -> [MLXArray] in
                let positive = stage.conditioning(tokens, start: Float(startSeconds), end: Float(end))
                let negative = negativeTokens.map { stage.conditioning($0, start: Float(startSeconds), end: Float(end)) }
                return [positive.context, positive.global] + (negative.map { [$0.context] } ?? [])
            }
            if !progress(0.02) {
                return nil
            }
            let steps = self.steps, guidance = self.guidance
            return try staging.use(pipelineStage) { pipeline -> MLXArray? in
                let keys = MLXRandom.split(key: MLXRandom.key(seed), into: steps + 2)
                let initial = MLXRandom.normal(pipeline.latentShape, key: keys[0])
                let encoded = try initialAudio.map {
                    try pipeline.encodedLatents($0, noise: MLXRandom.normal(pipeline.latentShape, key: keys[1]))
                }
                guard let latents = pipeline.latents(
                    context: conditioning[0], global: conditioning[1],
                    negativeContext: conditioning.count > 2 ? conditioning[2] : nil, steps: steps, guidance: guidance,
                    initial: initial, encoded: encoded,
                    noise: { MLXRandom.normal(pipeline.latentShape, key: keys[$0 + 2]) },
                    progress: { index, _ in progress(0.02 + 0.9 * Double(index + 1) / Double(steps)) }) else {
                    return nil
                }
                let waveform = pipeline.waveform(latents, start: startSeconds, end: end)
                eval(waveform)
                return waveform
            }
        }
    }

    /// A fine-tuning example from a consumer's own clip, as stable-audio-tools' dataset and pretransform
    /// prepare one: `waveform` `[samples, channels]` at ``sampleRate`` zero-padded or cropped to the latent
    /// window, encoded and drawn from the autoencoder's posterior, with the frames that hold the clip
    /// counted as `resize_padding_mask` counts them (rounding up); `prompt` through t5-base, masked; and the
    /// clip's place in its source, `startSeconds` and `totalSeconds` (the clip's own length where nil).
    ///
    /// Introduced in InferKit 0.4.0.
    public func trainingExample(waveform: MLXArray, prompt: String, startSeconds: Float = 0,
                                totalSeconds: Float? = nil, seed: UInt64 = 0) throws -> NFKMLXStableAudioExample {
        let window = transformerConfiguration.sampleSize * autoencoderConfiguration.hopLength
        let valid = min(waveform.dim(0), window)
        let validFrames = Int((Double(valid) * Double(transformerConfiguration.sampleSize) / Double(window)).rounded(.up))
        let tokens = self.prompt(prompt)
        return try staging.exclusively {
            let text = try staging.with(textStage) { stage -> MLXArray in
                let mask = tokens.maskArray
                return (stage.encoder(tokens.tokens, mask: mask) * mask.expandedDimensions(axis: -1).asType(.float32))[0]
            }
            let latents = try staging.with(pipelineStage) { pipeline -> MLXArray in
                try pipeline.encodedLatents(waveform, noise: MLXRandom.normal(pipeline.latentShape, key: MLXRandom.key(seed)))[0]
            }
            return NFKMLXStableAudioExample(latents: latents, validFrames: validFrames, text: text,
                                            startSeconds: startSeconds,
                                            totalSeconds: totalSeconds ?? Float(Double(waveform.dim(0)) / Double(sampleRate)))
        }
    }

    /// Generates a clip for `prompt` and writes it to `url` as a 16-bit stereo WAV file, clipped to
    /// `-1...1`. Returns the clip's duration in seconds.
    @objc(writeAudioForPrompt:negativePrompt:startSeconds:endSeconds:seed:toURL:error:)
    public func writeAudio(forPrompt prompt: String, negativePrompt: String?, startSeconds: Double,
                           endSeconds: Double, seed: UInt64, to url: URL) throws -> NSNumber {
        guard let clip = try audio(forPrompt: prompt, negativePrompt: negativePrompt, startSeconds: startSeconds,
                                   endSeconds: endSeconds, seed: seed) else { throw NFKMLXError.noOutput }
        try NFKMLXWaveFile.write(samples: clip.asType(.float32).clipped(-1, 1).reshaped([-1]).asArray(Float.self),
                                 sampleRate: sampleRate, channels: channelCount, to: url)
        return NSNumber(value: Double(clip.dim(0)) / Double(sampleRate))
    }
}

private extension MLXArray {
    func clipped(_ low: Float, _ high: Float) -> MLXArray { clip(self, min: low, max: high) }
}

extension NFKMLXStableAudioGenerator {

    /// The shortest clip a request asks for, the catalog's duration minimum.
    static let minimumRequestDuration = 0.5

    /// The request inputs the backend reads.
    static let backendInputKeys: Set<String> = [NFKInputPrompt, NFKInputNegativePrompt, NFKInputAudio]

    /// The request parameters the backend reads.
    static let backendParameterKeys: Set<String> = [
        NFKParameterDurationSeconds, NFKParameterSeed, NFKParameterSteps, NFKParameterGuidanceScale,
        NFKMLXStableAudioParameterKey.startSeconds,
    ]

    /// The generator behind the InferKit contract.
    ///
    /// @discussion The backend reads `NFKInputPrompt` (required), `NFKInputNegativePrompt`, and
    /// `NFKInputAudio` (a 16-bit PCM WAV `NFKAudioAsset` or `Data` at the model's sample rate, the clip to
    /// start from), and returns a stereo 44.1 kHz `NFKAudioAsset` under `NFKOutputAudio`. Parameters:
    /// `NFKParameterDurationSeconds` (10, from 0.5 seconds, the clip ending within ``maximumDuration``), the start
    /// under `NFKMLXStableAudioParameterKey.startSeconds` (0), `NFKParameterSeed` (0), `NFKParameterSteps` and
    /// `NFKParameterGuidanceScale` (the generator's ``steps`` and ``guidance``). A request's steps and
    /// guidance apply to that request only. Introduced in InferKit 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        NFKMLXGenerationBackend(identifier: Self.modelName, inputKeys: Self.backendInputKeys,
                                parameterKeys: Self.backendParameterKeys,
                                outputKeys: [NFKOutputAudio]) { [self] request, job in
            let prompt = try NFKMLXGenerationBackend.requiredText(request, NFKInputPrompt)
            let negative = NFKMLXGenerationBackend.text(request, NFKInputNegativePrompt)
            func parameter(_ key: String) -> NSNumber? { NFKMLXGenerationBackend.number(request, key) }
            let start = parameter(NFKMLXStableAudioParameterKey.startSeconds)?.doubleValue ?? 0
            let duration = parameter(NFKParameterDurationSeconds)?.doubleValue ?? 10
            guard duration >= Self.minimumRequestDuration else {
                throw NFKMLXError.unsupportedConfiguration("a clip runs at least \(Self.minimumRequestDuration) seconds")
            }
            let initialAudio = try Self.initialAudio(request, sampleRate: sampleRate)
            let (savedSteps, savedGuidance) = (steps, guidance)
            defer { (steps, guidance) = (savedSteps, savedGuidance) }
            steps = parameter(NFKParameterSteps)?.intValue ?? steps
            guidance = parameter(NFKParameterGuidanceScale)?.floatValue ?? guidance
            job.reportProgress(0)
            guard let clip = try audio(forPrompt: prompt, negativePrompt: negative, startSeconds: start,
                                       endSeconds: start + duration, initialAudio: initialAudio,
                                       seed: NFKMLXGenerationBackend.seed(request),
                                       progress: { fraction in
                                           job.reportProgress(fraction)
                                           return job.status != .cancelled
                                       }) else { return nil }
            let samples = clip.asType(.float32).clipped(-1, 1).reshaped([-1]).asArray(Float.self)
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("\(Self.modelName)-\(UUID().uuidString).wav")
            try NFKMLXWaveFile.write(samples: samples, sampleRate: sampleRate, channels: channelCount, to: url)
            let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(clip.dim(0)) / Double(sampleRate),
                                      sampleRate: Double(sampleRate), channelCount: channelCount)
            job.reportProgress(1)
            return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
        }
    }

    /// The waveform under `NFKInputAudio`, `[samples, channels]`, or nil where the request carries none. A
    /// clip at another sample rate is refused, as the reference refuses it.
    static func initialAudio(_ request: NFKInferenceRequest, sampleRate: Int) throws -> MLXArray? {
        guard let value = request.input(forKey: NFKInputAudio) else { return nil }
        let data: Data?
        if let asset = value as? NFKAudioAsset, let url = asset.fileURL {
            data = try Data(contentsOf: url)
        } else {
            data = value as? Data
        }
        guard let data, let file = NFKMLXWaveFile.readChannels(data) else { throw NFKMLXError.unsupportedInput }
        guard file.sampleRate == sampleRate else {
            throw NFKMLXError.unsupportedConfiguration("the initial audio is at \(file.sampleRate) Hz; the model reads \(sampleRate) Hz")
        }
        return MLXArray(file.samples).reshaped([-1, file.channels])
    }

    /// Assembles the model from a downloaded release directory, held as ``NFKMLXResidency/automatic``
    /// decides, and returns it behind the InferKit contract (``inferenceBackend()``). Introduced in InferKit
    /// 0.4.0.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try generator(directoryURL: directoryURL).inferenceBackend()
    }

    /// ``backend(directoryURL:)`` holding the stages as `residency` says. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:residency:error:)
    public static func backend(directoryURL: URL, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try generator(directoryURL: directoryURL, residency: residency).inferenceBackend()
    }

    /// Downloads the release (``releaseRepo`` when `repo` is nil) and returns the model behind the InferKit
    /// contract.
    ///
    /// @discussion The release is about 5.3 GB. The repository is gated: its license is accepted once on
    /// the Hub, and the token is `NFKHFHub.defaultAccessToken`. A file already in the cache is not fetched
    /// again. The call blocks on the network, so run it off the main and render threads. The weights are
    /// under the Stability AI Community License. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:residency:error:)
    public static func backend(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            try generator(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                          residency: residency).inferenceBackend()
        }
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:residency:)``. The handler runs
    /// on a background queue. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func backend(repo: String?, revision: String?, cacheDirectoryURL: URL?, residency: NFKMLXResidency,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL, residency: residency)
        }
    }

    /// Registers the backend under ``modelName``. The registry's URL is the release directory; nil is
    /// refused. Introduced in InferKit 0.4.0.
    @objc public static func register() {
        NFKMLXGenerationBackend.registerDirectoryModel(modelName) { try backend(directoryURL: $0) }
    }
}
