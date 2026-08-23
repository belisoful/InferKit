//
//  NFKMLXLatentDiffusionModels.swift
//  InferKitMLX
//
//  Two image-conditioned latent-diffusion models on `NFKMLXDiffusionBackend`, over the same real
//  networks the inpainter runs (`NFKMLXSDUNet` / `NFKMLXSDAutoencoder`, both at reference parity):
//
//  - Marigold (`marigold-depth`): image → depth. The UNet denoises a depth latent conditioned on the
//    image latent, and the decoded result is read as a grayscale map.
//  - The ×4 latent upscaler (`sd-x4-upscaler`): image → ×4 image. The UNet denoises a high-resolution
//    latent conditioned on the low-resolution image, with a noise level joining the timestep.
//

import Foundation
import InferKit
import MLX
import MLXNN

private final class NFKMLXLatentHolder: @unchecked Sendable {
    let model: NFKMLXLatentModel
    init(_ model: NFKMLXLatentModel) { self.model = model }
}

/// What the two models here share: a pipeline, a scale, and the batching the diffusion backend needs.
class NFKMLXLatentModel {
    let pipeline: NFKMLXSDPipeline
    let steps: Int
    let scale: Float
    let crossAttentionDimensions: Int

    init(unet: NFKMLXSDUNetConfiguration, vae: NFKMLXSDVAEConfiguration, steps: Int) {
        self.pipeline = NFKMLXSDPipeline(unet: unet, vae: vae)
        self.pipeline.train(false)
        self.steps = steps
        self.scale = vae.scaleFactor
        self.crossAttentionDimensions = unet.crossAttentionDimensions
    }

    var context: MLXArray { pipeline.batchedContext(dimensions: crossAttentionDimensions) }

    /// The image latent of a `[h, w, c]` plate in `0...1`, batched and scaled.
    func imageLatent(_ image: MLXArray) -> MLXArray {
        let batched = image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]]) * 2 - 1
        return pipeline.latent(of: batched, scale: scale)
    }

    func decoded(_ latent: MLXArray) -> MLXArray {
        let batched = latent.reshaped([1, latent.shape[0], latent.shape[1], latent.shape[2]])
        let image = clip((pipeline.image(from: batched, scale: scale) + 1) / 2, min: 0, max: 1)
        return image.reshaped([image.shape[1], image.shape[2], image.shape[3]])
    }
}

// MARK: Marigold depth

/// Marigold monocular depth as a latent-diffusion model.
@objc(NFKMLXMarigold)
public final class NFKMLXMarigold: NSObject {

    @objc public static let modelName = "marigold-depth"

    static func makeModel(steps: Int = 10) -> NFKMLXMarigoldModel {
        NFKMLXMarigoldModel(steps: steps)
    }

    /// Builds a Marigold depth backend from a single checkpoint holding both networks. A nil
    /// `weightsURL` builds random weights (`isReady` is true). Image under `NFKInputImage` → a depth
    /// map under `NFKOutputImage`. Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let model = makeModel()
        if let weightsURL {
            try model.pipeline.loadWeights(from: weightsURL)
        }
        return model.makeBackend()
    }

    /// Builds from the released diffusers layout: one checkpoint per network, plus the empty-prompt
    /// embedding Marigold conditions on.
    @objc(backendWithUNetWeightsURL:vaeWeightsURL:textContextURL:error:)
    public static func backend(unetWeightsURL: URL, vaeWeightsURL: URL,
                               textContextURL: URL?) throws -> any NFKInferenceBackend {
        let model = makeModel()
        try model.pipeline.loadWeights(unetURL: unetWeightsURL, vaeURL: vaeWeightsURL)
        if let textContextURL {
            try model.pipeline.loadTextContext(from: textContextURL)
        }
        return model.makeBackend()
    }

    /// Downloads the checkpoint from Hugging Face, then builds. Blocking
    /// on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath,
                                                revision: revision, cacheDirectoryURL: cacheDirectoryURL)
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

    /// Registers Marigold (`marigold-depth`) with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }
}

final class NFKMLXMarigoldModel: NFKMLXLatentModel {
    override init(unet: NFKMLXSDUNetConfiguration = .marigold,
                  vae: NFKMLXSDVAEConfiguration = .stableDiffusion, steps: Int) {
        super.init(unet: unet, vae: vae, steps: steps)
    }

    func makeBackend() -> NFKMLXDiffusionBackend {
        let holder = NFKMLXLatentHolder(self)
        var configuration = NFKDiffusionConfiguration(steps: steps)
        configuration.latentChannels = pipeline.vae.configuration.latentChannels

        return NFKMLXDiffusionBackend(
            identifier: NFKMLXMarigold.modelName, configuration: configuration,
            scheduler: NFKDDIMScheduler(predictionType: .epsilon),
            encode: { _, image, _ in
                guard let image else { throw NFKMLXError.unsupportedInput }
                let model = holder.model as! NFKMLXMarigoldModel
                let latent = model.imageLatent(image)
                let (h, w, c) = (latent.shape[1], latent.shape[2], latent.shape[3])
                return NFKDiffusionContext(conditioning: ["image": latent.reshaped([h, w, c])],
                                           width: w, height: h)
            },
            denoise: { latent, timestep, context, _ in
                let model = holder.model as! NFKMLXMarigoldModel
                guard let image = context.conditioning["image"] else { return latent }
                let batched = latent.reshaped([1, context.height, context.width,
                                               latent.shape[latent.ndim - 1]])
                let conditioned = image.reshaped([1, context.height, context.width, image.shape[2]])
                // Marigold conditions by concatenation: the depth latent then the image latent.
                let prediction = model.pipeline.unet(
                    concatenated([batched, conditioned], axis: 3),
                    timestep: MLXArray([Int32(timestep.train)]), context: model.context)
                return prediction.reshaped(latent.shape)
            },
            decode: { latent in (holder.model as! NFKMLXMarigoldModel).decoded(latent) })
    }
}

// MARK: SD ×4 latent upscaler

/// A Stable Diffusion latent upscaler (×4) as a latent-diffusion model.
@objc(NFKMLXSDUpscaler)
public final class NFKMLXSDUpscaler: NSObject {

    @objc public static let modelName = "sd-x4-upscaler"

    static func makeModel(steps: Int = 20, noiseLevel: Int = 20) -> NFKMLXSDUpscalerModel {
        NFKMLXSDUpscalerModel(steps: steps, noiseLevel: noiseLevel)
    }

    /// Builds an SD ×4 latent-upscaler backend from a single checkpoint holding both networks. A nil
    /// `weightsURL` builds random weights (`isReady` is true). Image under `NFKInputImage` → a ×4
    /// image. Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let model = makeModel()
        if let weightsURL {
            try model.pipeline.loadWeights(from: weightsURL)
        }
        return model.makeBackend()
    }

    /// Builds from the released diffusers layout. `noiseLevel` is the class the UNet conditions on;
    /// the release is trained over 0…349 and its own pipeline defaults to 20.
    @objc(backendWithUNetWeightsURL:vaeWeightsURL:textContextURL:noiseLevel:error:)
    public static func backend(unetWeightsURL: URL, vaeWeightsURL: URL, textContextURL: URL?,
                               noiseLevel: Int) throws -> any NFKInferenceBackend {
        let model = makeModel(noiseLevel: noiseLevel)
        try model.pipeline.loadWeights(unetURL: unetWeightsURL, vaeURL: vaeWeightsURL)
        if let textContextURL {
            try model.pipeline.loadTextContext(from: textContextURL)
        }
        return model.makeBackend()
    }

    /// Downloads the checkpoint from Hugging Face, then builds. Blocking
    /// on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath,
                                                revision: revision, cacheDirectoryURL: cacheDirectoryURL)
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

    /// Registers the ×4 latent upscaler (`sd-x4-upscaler`) with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }
}

final class NFKMLXSDUpscalerModel: NFKMLXLatentModel {
    let noiseLevel: Int

    init(unet: NFKMLXSDUNetConfiguration = .upscaler, vae: NFKMLXSDVAEConfiguration = .upscaler,
         steps: Int, noiseLevel: Int) {
        self.noiseLevel = noiseLevel
        super.init(unet: unet, vae: vae, steps: steps)
    }

    func makeBackend() -> NFKMLXDiffusionBackend {
        let holder = NFKMLXLatentHolder(self)
        var configuration = NFKDiffusionConfiguration(steps: steps)
        configuration.latentChannels = pipeline.vae.configuration.latentChannels

        return NFKMLXDiffusionBackend(
            identifier: NFKMLXSDUpscaler.modelName, configuration: configuration,
            scheduler: NFKDDIMScheduler(predictionType: .epsilon),
            encode: { _, image, _ in
                guard let image else { throw NFKMLXError.unsupportedInput }
                // The upscaler conditions on the low-resolution image itself, not on its latent, and
                // denoises a latent four times its size.
                let low = image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]]) * 2 - 1
                return NFKDiffusionContext(conditioning: ["low": low[0]],
                                           width: image.shape[1], height: image.shape[0])
            },
            denoise: { latent, timestep, context, _ in
                let model = holder.model as! NFKMLXSDUpscalerModel
                guard let low = context.conditioning["low"] else { return latent }
                let batched = latent.reshaped([1, context.height, context.width,
                                               latent.shape[latent.ndim - 1]])
                let conditioned = low.reshaped([1, context.height, context.width, low.shape[2]])
                let prediction = model.pipeline.unet(
                    concatenated([batched, conditioned], axis: 3),
                    timestep: MLXArray([Int32(timestep.train)]), context: model.context,
                    classLabel: MLXArray([Int32(model.noiseLevel)]))
                return prediction.reshaped(latent.shape)
            },
            decode: { latent in (holder.model as! NFKMLXSDUpscalerModel).decoded(latent) })
    }
}
