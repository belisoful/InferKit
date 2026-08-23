//
//  NFKMLXStableDiffusionInpaint.swift
//  InferKitMLX
//
//  Stable Diffusion inpainting on `NFKMLXDiffusionBackend`.
//
//  The networks are the real ones (`NFKMLXSDUNet` / `NFKMLXSDAutoencoder`, both at reference parity);
//  this file is the wiring that turns a plate and a mask into the nine-channel input the released
//  inpainting UNet takes, and the decoded result back into an image.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// The geometry and schedule of an inpainting pipeline.
public struct NFKMLXSDInpaintConfiguration: Sendable {
    public var unet: NFKMLXSDUNetConfiguration = .inpainting
    public var vae: NFKMLXSDVAEConfiguration = .stableDiffusion
    public var steps: Int = 20

    public var latentChannels: Int { vae.latentChannels }
    public var scaleFactor: Float { vae.scaleFactor }

    public init() {}

    /// The released SD 1.5 inpainting geometry, which the defaults spell out.
    public static let stableDiffusion15 = NFKMLXSDInpaintConfiguration()
}

/// Stable Diffusion inpainting, wired onto `NFKMLXDiffusionBackend`, and its registration.
@objc(NFKMLXStableDiffusionInpaint)
public final class NFKMLXStableDiffusionInpaint: NSObject {

    @objc public static let modelName = "sd-inpaint"

    static func makeModel(_ configuration: NFKMLXSDInpaintConfiguration = .stableDiffusion15)
        -> NFKMLXSDInpaintModel {
        NFKMLXSDInpaintModel(configuration)
    }

    /// Builds an SD-inpaint backend from a single checkpoint holding both networks — the layout
    /// `NFKMLXWeights.save` writes. A nil `weightsURL` builds random weights (`isReady` is
    /// true). Plate under `NFKInputImage`, mask under `NFKInputMask`.
    /// Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let model = makeModel()
        if let weightsURL {
            try model.pipeline.loadWeights(from: weightsURL)
        }
        return model.makeBackend()
    }

    /// Builds from the released diffusers layout: one checkpoint per network, plus the text embedding
    /// the trained UNet cross-attends to.
    ///
    /// - Parameters:
    ///   - unetWeightsURL: the release's `unet/` checkpoint, converted to safetensors.
    ///   - vaeWeightsURL: the release's `vae/` checkpoint, converted to safetensors.
    ///   - textContextURL: a safetensors holding a `[tokens, 768]` embedding under `context`. The
    ///     released model is trained with text conditioning, so an empty-prompt embedding — not
    ///     nothing — is what an unprompted run supplies.
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

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath,
                                                revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers SD inpainting (`sd-inpaint`) with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }
}

/// Owns the pipeline and builds the diffusion backend around it.
final class NFKMLXSDInpaintModel {
    let pipeline: NFKMLXSDPipeline
    let configuration: NFKMLXSDInpaintConfiguration

    init(_ configuration: NFKMLXSDInpaintConfiguration) {
        self.configuration = configuration
        self.pipeline = NFKMLXSDPipeline(unet: configuration.unet, vae: configuration.vae)
        self.pipeline.train(false)
    }

    /// VAE-encode into the conditioning, UNet as the denoiser, VAE-decode as the output. The backend
    /// runs the DDIM loop and the per-step inpaint compositing.
    func makeBackend() -> NFKMLXDiffusionBackend {
        let holder = NFKMLXSDInpaintHolder(self)
        var backendConfiguration = NFKDiffusionConfiguration(steps: configuration.steps)
        backendConfiguration.latentChannels = configuration.latentChannels

        return NFKMLXDiffusionBackend(
            identifier: NFKMLXStableDiffusionInpaint.modelName,
            configuration: backendConfiguration,
            scheduler: NFKDDIMScheduler(predictionType: .epsilon),
            encode: { _, image, mask in try holder.model.encode(image: image, mask: mask) },
            denoise: { latent, timestep, context, _ in
                holder.model.denoise(latent, timestep, context)
            },
            decode: { latent in holder.model.decode(latent) })
    }

    private var scale: Float { configuration.scaleFactor }

    // The diffusion backend works with unbatched `[h, w, c]` latents, so the context tensors are 3-D;
    // the networks run batched, and the batch dimension is added and dropped at those calls.
    func encode(image: MLXArray?, mask: MLXArray?) throws -> NFKDiffusionContext {
        guard let image, let mask else { throw NFKMLXError.unsupportedInput }
        let (height, width) = (image.shape[0], image.shape[1])
        // The autoencoder is trained on `-1...1`, where the image bridge delivers `0...1`.
        let batched = image.reshaped([1, height, width, image.shape[2]]) * 2 - 1
        let batchedMask = mask.reshaped([1, height, width, 1])

        let source = pipeline.latent(of: batched, scale: scale)
        let latentHeight = source.shape[1], latentWidth = source.shape[2]
        let channels = source.shape[3]
        // The masked-image latent is the plate with the hole blanked, encoded — not the plate's latent
        // with the hole blanked, which is a different tensor because the encoder is not local.
        let masked = pipeline.latent(of: batched * (1 - batchedMask), scale: scale)
            .reshaped([latentHeight, latentWidth, channels])
        let maskLatent = NFKMLXResample.resizeNearest(batchedMask, height: latentHeight,
                                                      width: latentWidth)
            .reshaped([latentHeight, latentWidth, 1])

        return NFKDiffusionContext(
            conditioning: ["masked": masked, "maskLatent": maskLatent],
            width: latentWidth, height: latentHeight,
            sourceLatent: source.reshaped([latentHeight, latentWidth, channels]), mask: maskLatent)
    }

    func denoise(_ latent: MLXArray, _ timestep: NFKDiffusionTimestep,
                 _ context: NFKDiffusionContext) -> MLXArray {
        guard let masked = context.conditioning["masked"],
              let maskLatent = context.conditioning["maskLatent"] else {
            return latent
        }
        let channels = latent.shape[latent.ndim - 1]
        let batched = latent.reshaped([1, context.height, context.width, channels])
        let batchedMasked = masked.reshaped([1, context.height, context.width, masked.shape[2]])
        let batchedMask = maskLatent.reshaped([1, context.height, context.width, 1])
        // The released UNet takes the noisy latent, then the mask, then the masked-image latent.
        let input = concatenated([batched, batchedMask, batchedMasked], axis: 3)
        let prediction = pipeline.unet(
            input, timestep: MLXArray([Int32(timestep.train)]),
            context: pipeline.batchedContext(dimensions: configuration.unet.crossAttentionDimensions))
        return prediction.reshaped(latent.shape)
    }

    func decode(_ latent: MLXArray) -> MLXArray {
        let batched = latent.reshaped([1, latent.shape[0], latent.shape[1], latent.shape[2]])
        let image = pipeline.image(from: batched, scale: scale)
        // Back to the `0...1` the image bridge expects.
        let clamped = clip((image + 1) / 2, min: 0, max: 1)
        return clamped.reshaped([clamped.shape[1], clamped.shape[2], clamped.shape[3]])
    }
}

private final class NFKMLXSDInpaintHolder: @unchecked Sendable {
    let model: NFKMLXSDInpaintModel
    init(_ model: NFKMLXSDInpaintModel) { self.model = model }
}
