//
//  NFKMLXFluxControlNetGenerator.swift
//  InferKitMLX
//
//  FLUX.1 steered by a ControlNet, from a downloaded diffusers FLUX release directory and a ControlNet
//  release directory: CLIP-L and T5-XXL run once per image, then the transformer, the ControlNet, and
//  the autoencoder, the two stages held as an `NFKMLXResidency` says.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// FLUX.1 text-to-image steered by a control image through a FLUX ControlNet.
///
/// @discussion The base release is the diffusers FLUX layout ``NFKMLXFlux`` reads. The ControlNet
/// release is a `FluxControlNetModel` directory (`config.json` and its weights). A base whose
/// transformer embeds guidance is FLUX.1 [dev]: it runs the dynamic-shift schedule over 28 steps with
/// a 512-token T5 sequence, which is how the FLUX ControlNets were trained. A base without it is
/// [schnell]: the four-step schedule with a 256-token sequence. A union ControlNet reads a control
/// type (``controlModeCount`` is non-zero); a single-type one reads none. The glue is measured against
/// diffusers' `FluxControlNetPipeline`.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXFluxControlNetGenerator)
public final class NFKMLXFluxControlNetGenerator: NSObject {

    /// A name for the model the factories produce. Introduced in InferKit 0.4.0.
    @objc public static let modelName = "flux.1-controlnet"

    /// The base repository the download factories read when none is named. It is gated: its license
    /// is accepted once on the Hub, and the token is `NFKHFHub.defaultAccessToken`. Introduced in InferKit 0.4.0.
    @objc public static let releaseRepo = "black-forest-labs/FLUX.1-dev"

    /// The ControlNet repository the download factories read when none is named. Introduced in InferKit 0.4.0.
    @objc public static let controlNetReleaseRepo = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"

    private let staging: NFKMLXStagedModel
    private let textStage: NFKMLXStage<NFKMLXFluxTextEncoder>
    private let pipelineStage: NFKMLXStage<NFKMLXFluxControlNetPipeline>

    /// The control types a union ControlNet reads, or 0 for a single-type ControlNet. Introduced in InferKit 0.4.0.
    @objc public let controlModeCount: Int

    /// The pixels one latent cell decodes to: 8 for the released autoencoder. Introduced in InferKit 0.4.0.
    @objc public let pixelsPerLatent: Int

    /// The denoising steps: 28 for a [dev] base, 4 for [schnell]. Introduced in InferKit 0.4.0.
    @objc public var steps: Int
    /// The guidance a guidance-distilled ([dev]) base embeds, 3.5 by the reference's default.
    /// Introduced in InferKit 0.4.0.
    @objc public var guidance: Float = 3.5
    /// The weight the ControlNet's residuals carry, 1 by the reference's default. Introduced in InferKit 0.4.0.
    @objc public var conditioningScale: Float = 1

    /// Whether the text encoders and the transformer stay loaded together between images.
    /// Introduced in InferKit 0.4.0.
    @objc public var holdsStagesResident: Bool { staging.resident }

    init(resident: Bool, controlModeCount: Int, steps: Int, pixelsPerLatent: Int = 8,
         loadTextEncoder: @escaping () throws -> NFKMLXFluxTextEncoder,
         loadPipeline: @escaping () throws -> NFKMLXFluxControlNetPipeline) {
        staging = NFKMLXStagedModel(resident: resident)
        textStage = NFKMLXStage(loadTextEncoder)
        pipelineStage = NFKMLXStage(loadPipeline)
        self.controlModeCount = controlModeCount
        self.steps = steps
        self.pixelsPerLatent = pixelsPerLatent
        super.init()
    }

    /// Loads both stages now, for a resident release.
    func loadResident() throws {
        try staging.use(textStage) { _ in }
        try staging.use(pipelineStage) { _ in }
    }

    var isHoldingTextEncoder: Bool { staging.exclusively { textStage.isHeld } }
    var isHoldingPipeline: Bool { staging.exclusively { pipelineStage.isHeld } }

    // MARK: Factories

    /// Assembles the model from a FLUX.1 release directory and a ControlNet release directory, holding
    /// it as ``NFKMLXResidency/automatic`` decides. Introduced in InferKit 0.4.0.
    @objc(generatorWithDirectoryURL:controlNetDirectoryURL:error:)
    public static func generator(directoryURL: URL, controlNetDirectoryURL: URL) throws -> NFKMLXFluxControlNetGenerator {
        try generator(directoryURL: directoryURL, controlNetDirectoryURL: controlNetDirectoryURL, residency: .automatic)
    }

    /// Assembles the model from a FLUX.1 release directory and a ControlNet release directory, holding
    /// it as `residency` says.
    ///
    /// @discussion CLIP-L loads as stored and T5-XXL at float32 (about 19 GB). The transformer, the
    /// ControlNet, and the autoencoder load as stored, the [dev] transformer at bfloat16 (about 24 GB).
    /// A machine that cannot hold the two stages together stages under ``NFKMLXResidency/automatic``.
    /// FLUX.1 has no routed experts, so ``NFKMLXResidency/paged`` holds it as
    /// ``NFKMLXResidency/staged`` does. Introduced in InferKit 0.4.0.
    @objc(generatorWithDirectoryURL:controlNetDirectoryURL:residency:error:)
    public static func generator(directoryURL: URL, controlNetDirectoryURL: URL,
                                 residency: NFKMLXResidency) throws -> NFKMLXFluxControlNetGenerator {
        let transformerDirectory = directoryURL.appendingPathComponent("transformer")
        let configuration = try NFKMLXFluxTransformerNet.configuration(
            fromHuggingFace: transformerDirectory.appendingPathComponent("config.json"))
        let controlConfiguration = try NFKMLXFluxControlNetNet.configuration(
            fromHuggingFace: controlNetDirectoryURL.appendingPathComponent("config.json"))
        let vaeURL = directoryURL.appendingPathComponent("vae/diffusion_pytorch_model.safetensors")
        let distilled = configuration.guidanceEmbeds
        let schedule: NFKMLXFlowMatchConfiguration = distilled ? .flux : .fluxSchnell
        let t5Context = NFKMLXFluxTextEncoder.t5Context(for: configuration)

        let fileBytes = { (url: URL) in
            (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
        }
        // CLIP-L loads as stored; T5-XXL loads at float32, twice its stored bfloat16.
        let encoderBytes = fileBytes(directoryURL.appendingPathComponent("text_encoder/model.safetensors"))
            + 2 * (try NFKMLXReleaseWeights.weightBytes(inDirectory: directoryURL.appendingPathComponent("text_encoder_2")))
        let pipelineBytes = try NFKMLXReleaseWeights.weightBytes(inDirectory: transformerDirectory)
            + NFKMLXReleaseWeights.weightBytes(inDirectory: controlNetDirectoryURL) + fileBytes(vaeURL)
        let plan = try NFKMLXResidencyBudget.plan(
            [NFKMLXStageFootprint(bytes: encoderBytes), NFKMLXStageFootprint(bytes: pipelineBytes)],
            residency: residency, budget: NFKMLXResidencyBudget.current())

        let generator = NFKMLXFluxControlNetGenerator(
            resident: plan.holdsStagesResident, controlModeCount: controlConfiguration.numMode ?? 0,
            steps: distilled ? 28 : 4,
            pixelsPerLatent: 1 << (NFKMLXSDVAEConfiguration.flux.blockChannels.count - 1),
            loadTextEncoder: { try NFKMLXFluxTextEncoder.textEncoder(directoryURL: directoryURL, t5Context: t5Context) },
            loadPipeline: {
                let transformer = NFKMLXFluxTransformerNet(configuration)
                try NFKMLXFluxTransformerNet.loadWeights(into: transformer, from: transformerDirectory,
                                                         precision: .checkpoint)
                let controlnet = NFKMLXFluxControlNetNet(controlConfiguration)
                try NFKMLXFluxControlNetNet.loadWeights(into: controlnet, from: controlNetDirectoryURL,
                                                        precision: .checkpoint)
                let vae = NFKMLXSDAutoencoder(configuration: .flux)
                try NFKMLXStableDiffusionModels.loadVAEWeights(into: vae, from: vaeURL, precision: .checkpoint)
                return NFKMLXFluxControlNetPipeline(transformer: transformer, controlnet: controlnet, vae: vae,
                                                    schedule: schedule)
            })
        if plan.holdsStagesResident {
            try generator.loadResident()
        }
        return generator
    }

    /// The files a ControlNet download fetches.
    static let controlNetComponents = [
        NFKMLXReleaseComponent(required: ["config.json"],
                               weights: ["diffusion_pytorch_model.safetensors",
                                         "diffusion_pytorch_model.safetensors.index.json"]),
    ]

    /// Downloads the base release (``releaseRepo`` when `repo` is nil) and the ControlNet
    /// (``controlNetReleaseRepo`` when `controlNetRepo` is nil), and assembles the model, holding it as
    /// `residency` says.
    ///
    /// @discussion `revision` names the base release's revision; the ControlNet is read at its main
    /// revision. The [dev] base is about 34 GB and the Union-Pro ControlNet about 6.6 GB. A file already
    /// in the cache is not fetched again. The call blocks on the network, so run it off the main and
    /// render threads. Introduced in InferKit 0.4.0.
    @objc(generatorWithRepo:controlNetRepo:revision:cacheDirectoryURL:residency:error:)
    public static func generator(repo: String?, controlNetRepo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency) throws -> NFKMLXFluxControlNetGenerator {
        let base = try NFKMLXFlux.releaseDirectory(repo: repo ?? releaseRepo, revision: revision,
                                                   cacheDirectoryURL: cacheDirectoryURL)
        let control = try NFKMLXReleaseDownload.directory(repo: controlNetRepo ?? controlNetReleaseRepo, revision: nil,
                                                          cacheDirectoryURL: cacheDirectoryURL,
                                                          components: controlNetComponents)
        return try generator(directoryURL: base, controlNetDirectoryURL: control, residency: residency)
    }

    /// The asynchronous form of ``generator(repo:controlNetRepo:revision:cacheDirectoryURL:residency:)``.
    /// The handler runs on a background queue. Introduced in InferKit 0.4.0.
    @objc(generatorWithRepo:controlNetRepo:revision:cacheDirectoryURL:residency:completionHandler:)
    public static func generator(repo: String?, controlNetRepo: String?, revision: String?, cacheDirectoryURL: URL?,
                                 residency: NFKMLXResidency,
                                 completionHandler: @escaping (NFKMLXFluxControlNetGenerator?, Error?) -> Void) {
        NFKMLXReleaseDownload.asyncValue(completionHandler) {
            try generator(repo: repo, controlNetRepo: controlNetRepo, revision: revision,
                          cacheDirectoryURL: cacheDirectoryURL, residency: residency)
        }
    }

    // MARK: Generation

    /// Generates an image for `prompt` steered by `controlImage`, `[height, width, 3]` RGB in `[0, 1]`.
    ///
    /// @discussion `controlImage` is `[1, height, width, 3]` in `[-1, 1]`. `width` and `height` are
    /// positive multiples of twice ``pixelsPerLatent`` (16 for the release). A union ControlNet requires `controlMode` in `0 ..< controlModeCount`;
    /// a single-type ControlNet refuses one. Introduced in InferKit 0.4.0.
    public func image(forPrompt prompt: String, controlImage: MLXArray, controlMode: Int?, width: Int,
                      height: Int, seed: UInt64 = 0) throws -> MLXArray {
        let multiple = 2 * pixelsPerLatent
        guard width > 0, height > 0, width % multiple == 0, height % multiple == 0 else {
            throw NFKMLXError.unsupportedConfiguration("an image's sides are positive multiples of \(multiple)")
        }
        guard controlImage.ndim == 4, controlImage.dim(1) == height, controlImage.dim(2) == width else {
            throw NFKMLXError.unsupportedConfiguration("the control image is [1, height, width, 3] at the output size")
        }
        if controlModeCount > 0 {
            guard let controlMode, (0 ..< controlModeCount).contains(controlMode) else {
                throw NFKMLXError.unsupportedConfiguration(
                    "the union ControlNet reads a control mode in 0..<\(controlModeCount)")
            }
        } else if controlMode != nil {
            throw NFKMLXError.unsupportedConfiguration("a single-type ControlNet reads no control mode")
        }
        return try staging.exclusively {
            let encoded = try staging.with(textStage) { encoder -> [MLXArray] in
                let (embeds, pooled) = encoder.encode(prompt: prompt)
                return [embeds, pooled]
            }
            return try staging.with(pipelineStage) { pipeline in
                let image = pipeline.generate(promptEmbeds: encoded[0], pooled: encoded[1], controlImage: controlImage,
                                              controlnetScale: conditioningScale, controlnetMode: controlMode,
                                              latentHeight: height / pixelsPerLatent, latentWidth: width / pixelsPerLatent, steps: steps,
                                              guidance: guidance, seed: seed)
                return clip(image[0].asType(.float32) / 2 + 0.5, min: 0, max: 1)
            }
        }
    }

    /// Generates an image for `prompt` steered by `controlImage`, resized to `width` × `height`, and
    /// returns it as a `CGImage`. `controlMode` is the union ControlNet's control type, or nil for a
    /// single-type ControlNet. Introduced in InferKit 0.4.0.
    @objc(imageForPrompt:controlImage:controlMode:width:height:seed:error:)
    public func cgImage(forPrompt prompt: String, controlImage: CGImage, controlMode: NSNumber?, width: Int,
                        height: Int, seed: UInt64) throws -> CGImage {
        let control = try NFKMLXControlImage.pixels(controlImage, width: width, height: height)
        let array = try image(forPrompt: prompt, controlImage: control, controlMode: controlMode?.intValue,
                              width: width, height: height, seed: seed)
        return try NFKMLXImageBridge.cgImage(from: array, options: NFKMLXImageOptions())
    }
}
