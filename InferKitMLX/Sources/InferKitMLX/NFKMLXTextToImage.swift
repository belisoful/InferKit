//
//  NFKMLXTextToImage.swift
//  InferKitMLX
//
//  Stable Diffusion text-to-image, on `NFKMLXDiffusionBackend`.
//
//  Every part is already in this package and at reference parity: `NFKMLXSDTextEncoderNet` encodes the
//  prompt, `NFKMLXSDUNet` predicts the noise, `NFKDDIMScheduler` samples, `NFKMLXSDAutoencoder` decodes.
//  This file is the wiring — a prompt becomes token ids, the ids become a conditioning sequence, and
//  the loop runs with classifier-free guidance.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// The geometry and schedule of a text-to-image release.
public struct NFKMLXSDTextToImageConfiguration: Sendable {
    public var unet: NFKMLXSDUNetConfiguration = NFKMLXSDUNetConfiguration()
    public var vae: NFKMLXSDVAEConfiguration = .stableDiffusion
    public var textEncoder: NFKMLXSDTextEncoderConfiguration = .stableDiffusion15
    /// A second text tower, for the releases that cross-attend to two. Its sequence joins the first's
    /// along the feature axis and it supplies the pooled embedding.
    public var secondaryTextEncoder: NFKMLXSDTextEncoderConfiguration?
    /// The latent side the release is trained at. 64 is a 512-pixel image through an eight-fold
    /// autoencoder.
    public var sampleSize: Int = 64
    public var steps: Int = 20
    public var guidanceScale: Float = 7.5
    public var predictionType: NFKDiffusionPredictionType = .epsilon
    public var timestepSpacing: NFKDiffusionTimestepSpacing = .leading
    /// The reference's `force_zeros_for_empty_prompt`. Where a release sets it, a request carrying NO
    /// negative prompt conditions on zeros rather than on the embedding of an empty sentence — the two
    /// differ, and SDXL is trained against the first. A request that supplies an empty string is
    /// asking for that sentence, and gets it, which is the distinction the reference draws too.
    public var zerosForEmptyNegativePrompt: Bool = false

    public var latentChannels: Int { vae.latentChannels }
    public var scaleFactor: Float { vae.scaleFactor }
    /// How many pixels one latent cell covers: the autoencoder halves at every level but the first.
    public var latentStride: Int { 1 << (vae.blockChannels.count - 1) }

    public init() {}

    /// Stable Diffusion 1.5, which the defaults spell out.
    public static let stableDiffusion15 = NFKMLXSDTextToImageConfiguration()

    /// Stable Diffusion 2.1 base: the same structure at a wider cross-attention, with the
    /// transformer's ends projected by a linear layer, conditioned on the OpenCLIP tower.
    public static let stableDiffusion21: NFKMLXSDTextToImageConfiguration = {
        var c = NFKMLXSDTextToImageConfiguration()
        c.unet.crossAttentionDimensions = 1024
        c.unet.attentionHeads = [5, 10, 20, 20]
        c.unet.usesLinearProjection = true
        c.textEncoder = .stableDiffusion21
        return c
    }()

    /// Stable Diffusion 2.1 at 768, whose model predicts the velocity rather than the noise.
    public static let stableDiffusion21V: NFKMLXSDTextToImageConfiguration = {
        var c = NFKMLXSDTextToImageConfiguration.stableDiffusion21
        c.predictionType = .vPrediction
        c.sampleSize = 96
        return c
    }()

    /// SDXL-Turbo: a distilled few-step release. Its schedule counts down from the end of the training
    /// range, and it is trained to run without classifier-free guidance.
    public static let sdxlTurbo: NFKMLXSDTextToImageConfiguration = {
        var c = NFKMLXSDTextToImageConfiguration()
        c.unet = .sdxl
        c.vae.scaleFactor = 0.13025
        c.textEncoder = .sdxlPrimary
        c.secondaryTextEncoder = .sdxlSecondary
        c.steps = 1
        c.guidanceScale = 0
        c.timestepSpacing = .trailing
        c.zerosForEmptyNegativePrompt = true
        return c
    }()

    /// A small configuration for offline structure and round-trip tests.
    public static let tiny: NFKMLXSDTextToImageConfiguration = {
        var c = NFKMLXSDTextToImageConfiguration()
        c.unet.blockChannels = [32, 64]
        c.unet.attends = [true, false]
        c.unet.attentionHeads = [2, 2]
        c.unet.layersPerBlock = 1
        c.unet.onlyCrossAttention = [false, false]
        c.unet.normalizationGroups = 8
        c.unet.crossAttentionDimensions = 32
        c.vae.blockChannels = [16, 32]
        c.vae.layersPerBlock = 1
        c.vae.normalizationGroups = 8
        c.textEncoder = .tiny
        c.sampleSize = 8
        c.steps = 2
        return c
    }()
}

/// Where a release's files sit. A downloaded release is a directory tree, so ``init(directoryURL:)``
/// resolves the usual one; a caller holding the files elsewhere names them individually.
public struct NFKMLXSDReleaseFiles: Sendable {
    public var unet: URL
    public var vae: URL
    public var textEncoder: URL
    public var tokenizer: URL
    /// The second tower's weights and vocabulary, for a release that carries one.
    public var secondaryTextEncoder: URL?
    public var secondaryTokenizer: URL?

    public init(unet: URL, vae: URL, textEncoder: URL, tokenizer: URL,
                secondaryTextEncoder: URL? = nil, secondaryTokenizer: URL? = nil) {
        self.unet = unet
        self.vae = vae
        self.textEncoder = textEncoder
        self.tokenizer = tokenizer
        self.secondaryTextEncoder = secondaryTextEncoder
        self.secondaryTokenizer = secondaryTokenizer
    }

    /// The released directory layout: `unet/`, `vae/`, `text_encoder/`, `tokenizer/`, and — for a
    /// two-tower release — `text_encoder_2/` and `tokenizer_2/`. A release that publishes only
    /// half-precision weights names them `.fp16.safetensors`, so both spellings are tried.
    public init(directoryURL: URL) throws {
        func weights(_ folder: String, _ base: String) throws -> URL {
            let directory = directoryURL.appendingPathComponent(folder)
            for name in ["\(base).safetensors", "\(base).fp16.safetensors"] {
                let candidate = directory.appendingPathComponent(name)
                if FileManager.default.fileExists(atPath: candidate.path) {
                    return candidate
                }
            }
            throw NFKMLXError.weightsMismatch("the release carries no \(folder)/\(base).safetensors")
        }
        unet = try weights("unet", "diffusion_pytorch_model")
        vae = try weights("vae", "diffusion_pytorch_model")
        textEncoder = try weights("text_encoder", "model")
        tokenizer = directoryURL.appendingPathComponent("tokenizer")
        let second = directoryURL.appendingPathComponent("text_encoder_2")
        if FileManager.default.fileExists(atPath: second.path) {
            secondaryTextEncoder = try weights("text_encoder_2", "model")
            secondaryTokenizer = directoryURL.appendingPathComponent("tokenizer_2")
        }
    }
}

/// Stable Diffusion text-to-image.
///
/// A request with no image input runs text-to-image; a request carrying a `CGImage` under
/// `NFKInputImage` runs image-to-image, with `NFKParameterStrength` controlling how much of the source
/// survives. `NFKInputPrompt`, `NFKInputNegativePrompt`, `NFKParameterSteps`,
/// `NFKParameterGuidanceScale`, `NFKParameterSeed`, and `NFKParameterWidth` / `NFKParameterHeight`
/// shape the run. The result carries the generated image under `NFKOutputImage`.
/// Run inference off the render thread.
@objc(NFKMLXTextToImage)
public final class NFKMLXTextToImage: NSObject {

    /// The identifier a backend built here reports. Text-to-image takes a whole release rather than a
    /// single checkpoint, so it has no `NFKMLXModelRegistry` entry — the registry's factory signature
    /// is one weights URL.
    @objc public static let modelName = "stable-diffusion"

    /// Builds a text-to-image backend from a release's own files.
    public static func backend(configuration: NFKMLXSDTextToImageConfiguration,
                               files: NFKMLXSDReleaseFiles,
                               precision: NFKMLXWeightPrecision = .float32) throws -> any NFKInferenceBackend {
        try NFKMLXSDTextToImageModel(configuration: configuration, files: files,
                                     precision: precision).makeBackend()
    }

    /// Builds a text-to-image backend from a downloaded release directory.
    ///
    /// A release published only in half precision loads at `.float32` by default, which doubles what it
    /// occupies; pass `.checkpoint` to run it as published.
    public static func backend(configuration: NFKMLXSDTextToImageConfiguration,
                               directoryURL: URL,
                               precision: NFKMLXWeightPrecision = .float32) throws -> any NFKInferenceBackend {
        try backend(configuration: configuration,
                    files: NFKMLXSDReleaseFiles(directoryURL: directoryURL), precision: precision)
    }

    /// Builds a text-to-image backend from a downloaded release directory, choosing the release by
    /// name — the Objective-C path, where the configuration struct cannot go.
    ///
    /// A release published only in half precision loads as published; a Swift caller passing
    /// `precision` chooses. Run inference off the render thread.
    @objc(backendWithModel:directoryURL:error:)
    public static func backend(model: NFKMLXStableDiffusionModel,
                               directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(configuration: NFKMLXStableDiffusionRelease(model).configuration,
                    directoryURL: directoryURL, precision: .checkpoint)
    }

    /// Builds a text-to-image backend with random weights, for a structure or round-trip test. The
    /// output is noise; a released checkpoint is what makes it an image.
    static func backend(configuration: NFKMLXSDTextToImageConfiguration) -> any NFKInferenceBackend {
        NFKMLXSDTextToImageModel(configuration: configuration).makeBackend()
    }
}

/// Owns the networks and the tokenizer, and builds the diffusion backend around them.
final class NFKMLXSDTextToImageModel {

    let configuration: NFKMLXSDTextToImageConfiguration
    let pipeline: NFKMLXSDPipeline
    let textEncoder: NFKMLXSDTextEncoderNet
    let tokenizer: NFKMLXSDPromptTokenizer?
    let secondaryTextEncoder: NFKMLXSDTextEncoderNet?
    let secondaryTokenizer: NFKMLXSDPromptTokenizer?

    /// A starting latent in place of the loop's own seeded noise, for reproducing a run made by
    /// another implementation. See ``NFKDiffusionContext/initialLatent``.
    var startLatent: MLXArray?

    /// An untrained model, for structure and round-trip tests. Without a tokenizer, a prompt encodes
    /// as the empty sequence.
    init(configuration: NFKMLXSDTextToImageConfiguration) {
        self.configuration = configuration
        pipeline = NFKMLXSDPipeline(unet: configuration.unet, vae: configuration.vae)
        pipeline.train(false)
        textEncoder = NFKMLXSDTextEncoderNet(configuration: configuration.textEncoder)
        textEncoder.train(false)
        tokenizer = nil
        secondaryTextEncoder = configuration.secondaryTextEncoder.map {
            let net = NFKMLXSDTextEncoderNet(configuration: $0)
            net.train(false)
            return net
        }
        secondaryTokenizer = nil
    }

    init(configuration: NFKMLXSDTextToImageConfiguration, files: NFKMLXSDReleaseFiles,
         precision: NFKMLXWeightPrecision = .float32) throws {
        self.configuration = configuration
        pipeline = NFKMLXSDPipeline(unet: configuration.unet, vae: configuration.vae)
        try pipeline.loadWeights(unetURL: files.unet, vaeURL: files.vae, precision: precision)
        pipeline.train(false)
        textEncoder = try NFKMLXSDTextEncoder.net(configuration: configuration.textEncoder,
                                                  weightsURL: files.textEncoder, precision: precision)
        tokenizer = try NFKMLXSDPromptTokenizer(directoryURL: files.tokenizer)

        guard let secondary = configuration.secondaryTextEncoder else {
            secondaryTextEncoder = nil
            secondaryTokenizer = nil
            return
        }
        guard let weightsURL = files.secondaryTextEncoder, let vocabularyURL = files.secondaryTokenizer else {
            throw NFKMLXError.weightsMismatch("this release cross-attends to two text towers, and the "
                                              + "second one's files are missing")
        }
        secondaryTextEncoder = try NFKMLXSDTextEncoder.net(configuration: secondary,
                                                            weightsURL: weightsURL, precision: precision)
        secondaryTokenizer = try NFKMLXSDPromptTokenizer(directoryURL: vocabularyURL)
    }

    func makeBackend(identifier: String = NFKMLXTextToImage.modelName) -> NFKMLXDiffusionBackend {
        let holder = NFKMLXSDTextToImageHolder(self)
        var backendConfiguration = NFKDiffusionConfiguration(steps: configuration.steps,
                                                             guidanceScale: configuration.guidanceScale)
        backendConfiguration.latentChannels = configuration.latentChannels

        return NFKMLXDiffusionBackend(
            identifier: identifier,
            configuration: backendConfiguration,
            scheduler: NFKDDIMScheduler(predictionType: configuration.predictionType,
                                        spacing: configuration.timestepSpacing),
            encode: { request, image, _ in try holder.model.encode(request: request, image: image) },
            denoise: { latent, timestep, context, guidance in
                holder.model.denoise(latent, timestep, context, guidance)
            },
            decode: { latent in holder.model.decode(latent) })
    }

    private var scale: Float { configuration.scaleFactor }

    /// The conditioning for a prompt: the tokens padded to each tower's context length, encoded, and —
    /// where a release carries two towers — run together along the feature axis. The pooled embedding
    /// comes from the second tower, which is the only one that carries a projection.
    ///
    /// A model built without a tokenizer encodes the empty sequence, which is what a structure test
    /// needs and what a real run must not do.
    func embedding(of prompt: String) -> NFKSDTextEmbedding {
        func tokens(_ tokenizer: NFKMLXSDPromptTokenizer?, _ length: Int) -> [Int] {
            tokenizer?.tokens(for: prompt, contextLength: length) ?? Array(repeating: 0, count: length)
        }
        let primary = textEncoder.encode(tokens(tokenizer, configuration.textEncoder.contextLength))
        guard let secondaryTextEncoder, let configured = configuration.secondaryTextEncoder else {
            return primary
        }
        let secondary = secondaryTextEncoder.encode(tokens(secondaryTokenizer, configured.contextLength))
        return NFKSDTextEmbedding(hidden: concatenated([primary.hidden, secondary.hidden], axis: -1),
                                  pooled: secondary.pooled)
    }

    // The diffusion backend works with unbatched `[h, w, c]` latents, so the tensors it holds are 3-D;
    // the networks run batched, and the batch dimension is added and dropped at those calls.
    func encode(request: NFKInferenceRequest, image: MLXArray?) throws -> NFKDiffusionContext {
        let conditional = embedding(of: request.prompt ?? "")
        let unconditional = configuration.zerosForEmptyNegativePrompt && request.negativePrompt == nil
            ? NFKSDTextEmbedding(hidden: MLXArray.zeros(like: conditional.hidden),
                                 pooled: conditional.pooled.map { MLXArray.zeros(like: $0) })
            : embedding(of: request.negativePrompt ?? "")
        var conditioning = ["context": conditional.hidden, "uncontext": unconditional.hidden]
        if let pooled = conditional.pooled, let unpooled = unconditional.pooled {
            conditioning["pooled"] = pooled
            conditioning["unpooled"] = unpooled
        }

        guard let image else {
            let (height, width) = latentSize(for: request)
            conditioning["timeIds"] = timeIds(height: height, width: width)
            return NFKDiffusionContext(conditioning: conditioning, width: width, height: height,
                                       initialLatent: startLatent)
        }
        // The autoencoder is trained on `-1...1`, where the image bridge delivers `0...1`.
        let batched = image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]]) * 2 - 1
        let source = pipeline.latent(of: batched, scale: scale)
        let (height, width, channels) = (source.shape[1], source.shape[2], source.shape[3])
        conditioning["timeIds"] = timeIds(height: height, width: width)
        return NFKDiffusionContext(conditioning: conditioning, width: width, height: height,
                                   sourceLatent: source.reshaped([height, width, channels]))
    }

    /// The latent grid a text-to-image request runs at: the release's own size, or a caller's pixel
    /// size divided by the autoencoder's stride and rounded to a whole latent cell.
    private func latentSize(for request: NFKInferenceRequest) -> (height: Int, width: Int) {
        func side(_ key: String) -> Int {
            guard let pixels = (request.parameter(forKey: key) as? NSNumber)?.intValue, pixels > 0 else {
                return configuration.sampleSize
            }
            return max(pixels / configuration.latentStride, 1)
        }
        return (side(NFKParameterHeight), side(NFKParameterWidth))
    }

    /// The reference's `time_ids`: the original size, the crop's top-left corner, and the target size,
    /// each in pixels. A plain generation crops nothing and targets what it renders.
    private func timeIds(height: Int, width: Int) -> MLXArray {
        let stride = configuration.latentStride
        let pixels = [height * stride, width * stride, 0, 0, height * stride, width * stride]
        return MLXArray(pixels.map { Float($0) }).reshaped([1, 6])
    }

    /// One denoising step. Above a guidance of 1 the conditional and unconditional predictions run as
    /// one batch of two, which is what makes classifier-free guidance one forward pass rather than two.
    func denoise(_ latent: MLXArray, _ timestep: NFKDiffusionTimestep,
                 _ context: NFKDiffusionContext, _ guidanceScale: Float) -> MLXArray {
        guard let conditional = context.conditioning["context"] else { return latent }
        let channels = latent.shape[latent.ndim - 1]
        let batched = latent.reshaped([1, context.height, context.width, channels])
        let step = MLXArray([Int32(timestep.train)])

        guard guidanceScale > 1, let unconditional = context.conditioning["uncontext"] else {
            return pipeline.unet(batched, timestep: step, context: conditional,
                                 added: added(context, key: "pooled", batch: 1)).reshaped(latent.shape)
        }
        // The reference orders the batch unconditional first, and reads the pair back the same way.
        let prediction = pipeline.unet(concatenated([batched, batched], axis: 0), timestep: step,
                                       context: concatenated([unconditional, conditional], axis: 0),
                                       added: added(context, key: "unpooled", other: "pooled", batch: 2))
        let guided = prediction[0] + guidanceScale * (prediction[1] - prediction[0])
        return guided.reshaped(latent.shape)
    }

    /// The pooled embedding and size descriptor a two-tower release conditions on, batched to match the
    /// latent. A release without an addition embedding has none, and the UNet ignores what it is given.
    private func added(_ context: NFKDiffusionContext, key: String, other: String? = nil,
                       batch: Int) -> NFKSDAddedConditioning? {
        guard configuration.unet.additionEmbedding != nil,
              let first = context.conditioning[key],
              let timeIds = context.conditioning["timeIds"] else {
            return nil
        }
        let pooled = other.flatMap { context.conditioning[$0] }.map { concatenated([first, $0], axis: 0) }
            ?? first
        return NFKSDAddedConditioning(pooled: pooled,
                                      timeIds: repeated(timeIds, count: batch, axis: 0))
    }

    func decode(_ latent: MLXArray) -> MLXArray {
        let batched = latent.reshaped([1, latent.shape[0], latent.shape[1], latent.shape[2]])
        let image = pipeline.image(from: batched, scale: scale)
        // Back to the `0...1` the image bridge expects.
        let clamped = clip((image + 1) / 2, min: 0, max: 1)
        return clamped.reshaped([clamped.shape[1], clamped.shape[2], clamped.shape[3]])
    }
}

private final class NFKMLXSDTextToImageHolder: @unchecked Sendable {
    let model: NFKMLXSDTextToImageModel
    init(_ model: NFKMLXSDTextToImageModel) { self.model = model }
}
