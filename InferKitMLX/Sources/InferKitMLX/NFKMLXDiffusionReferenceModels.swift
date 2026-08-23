//
//  NFKMLXDiffusionReferenceModels.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import InferKit
import MLX

/// Reference diffusion models registered with `NFKMLXModelRegistry`, so an Objective-C consumer can
/// build and run an MLX diffusion pipeline by name (upscaler, depth, inpainting).
///
/// Each one wires `NFKMLXDiffusionBackend` — encode, denoise, decode, scheduler — for a real task and
/// I/O shape. To keep them runnable in CI without multi-gigabyte weights, the denoise closure is an
/// *oracle*: it returns the epsilon that drives the DDIM loop to a target derived from the input, so
/// the full sampler runs (progress, cancellation, image bridge) and the output is deterministic. A
/// real integration keeps the same wiring and replaces the oracle with a trained UNet forward (and
/// usually a VAE encode/decode in `decode`).
// These pipelines set `setsAlphaToOne` where the released models leave it false. That flag decides what
// signal ratio the FINAL step denoises against: the released Stable Diffusion schedulers use the ratio
// at training step 0, which leaves a little noise in the result, while 1 denoises completely. A real
// model's last epsilon is small enough that the difference does not show; these stand-ins drive the loop
// to an exact target, so it does, and the exact target is the whole point of them.
extension NFKMLXReferenceModels {

    /// A 2× diffusion upscaler under `"diffusion-upscaler"`. Input image → an image twice the size.
    @objc public static func registerDiffusionUpscaler() {
        NFKMLXModelRegistry.register(name: "diffusion-upscaler") { _ in
            var configuration = NFKDiffusionConfiguration()
            configuration.steps = 8
            let scale = 2
            return NFKMLXDiffusionBackend(
                identifier: "diffusion-upscaler",
                configuration: configuration,
                scheduler: NFKDDIMScheduler(predictionType: .epsilon, setsAlphaToOne: true),
                encode: { _, image, _ in
                    guard let image else { throw NFKMLXError.unsupportedInput }
                    let target = nearestUpscale(image, scale: scale)
                    return NFKDiffusionContext(conditioning: ["target": target],
                                               width: image.shape[1] * scale,
                                               height: image.shape[0] * scale)
                },
                denoise: oracleDenoise,
                decode: { clip($0, min: 0, max: 1) })
        }
    }

    /// A Marigold-style depth estimator under `"diffusion-depth"`. Input image → a grayscale depth map.
    @objc public static func registerDiffusionDepth() {
        NFKMLXModelRegistry.register(name: "diffusion-depth") { _ in
            var configuration = NFKDiffusionConfiguration()
            configuration.steps = 8
            return NFKMLXDiffusionBackend(
                identifier: "diffusion-depth",
                configuration: configuration,
                scheduler: NFKDDIMScheduler(predictionType: .epsilon, setsAlphaToOne: true),
                encode: { _, image, _ in
                    guard let image else { throw NFKMLXError.unsupportedInput }
                    let channels = split(image, parts: 3, axis: 2)      // [R, G, B], each [H, W, 1]
                    let luminance = channels[0] * 0.299 + channels[1] * 0.587 + channels[2] * 0.114
                    let depth = concatenated([luminance, luminance, luminance], axis: 2)
                    return NFKDiffusionContext(conditioning: ["target": depth],
                                               width: image.shape[1], height: image.shape[0])
                },
                denoise: oracleDenoise,
                decode: { clip($0, min: 0, max: 1) })
        }
    }

    /// An inpainter under `"diffusion-inpaint"`. Plate under `NFKInputImage` + a mask under
    /// `NFKInputMask` (white regenerates, black is kept) → the masked region regenerated, the rest
    /// preserved.
    @objc public static func registerDiffusionInpainter() {
        NFKMLXModelRegistry.register(name: "diffusion-inpaint") { _ in
            var configuration = NFKDiffusionConfiguration()
            configuration.steps = 8
            configuration.strength = 1
            return NFKMLXDiffusionBackend(
                identifier: "diffusion-inpaint",
                configuration: configuration,
                scheduler: NFKDDIMScheduler(predictionType: .epsilon, setsAlphaToOne: true),
                encode: { _, image, mask in
                    guard let image, let mask else { throw NFKMLXError.unsupportedInput }
                    let fill = image * 0 + 0.5                          // flat gray stand-in fill
                    let target = mask * fill + (1 - mask) * image
                    return NFKDiffusionContext(conditioning: ["target": target],
                                               width: image.shape[1], height: image.shape[0],
                                               sourceLatent: image, mask: mask)
                },
                denoise: oracleDenoise,
                decode: { clip($0, min: 0, max: 1) })
        }
    }

    /// A ControlNet-style pipeline under `"diffusion-controlnet"`, showing how a control map conditions
    /// generation without a full Stable Diffusion reimplementation. A control image under
    /// `NFKInputControl` (edges, depth, pose) is read from the request, bridged, and carried in
    /// `context.conditioning["control"]`. A real integration brings a ControlNet-capable UNet and, in
    /// its `denoise` closure, runs the control network on `conditioning["control"]` and adds the
    /// resulting residuals to the UNet's blocks; the backend still owns the sampler loop, guidance,
    /// progress, and the image bridge. Here the oracle `denoise` steers the output toward the control
    /// map so the conditioning is visibly in effect.
    @objc public static func registerControlNet() {
        NFKMLXModelRegistry.register(name: "diffusion-controlnet") { _ in
            var configuration = NFKDiffusionConfiguration()
            configuration.steps = 8
            return NFKMLXDiffusionBackend(
                identifier: "diffusion-controlnet",
                configuration: configuration,
                scheduler: NFKDDIMScheduler(predictionType: .epsilon, setsAlphaToOne: true),
                encode: { request, _, _ in
                    guard let controlValue = request.input(forKey: NFKInputControl) else {
                        throw NFKMLXError.unsupportedInput
                    }
                    let control = try NFKMLXImageBridge.tensor(from: controlValue, channels: 3,
                                                               colorSpace: CGColorSpaceCreateDeviceRGB())
                    // "control" is the slot a real ControlNet denoise reads; "target" drives the oracle.
                    return NFKDiffusionContext(conditioning: ["control": control, "target": control],
                                               width: control.shape[1], height: control.shape[0])
                },
                denoise: oracleDenoise,
                decode: { clip($0, min: 0, max: 1) })
        }
    }

    // MARK: Reference forward and image ops

    /// The epsilon that steers the DDIM loop toward `context.conditioning["target"]`. Stands in for a
    /// trained UNet forward; a real model reads the latent, timestep, and conditioning instead.
    static let oracleDenoise: NFKMLXDiffusionBackend.Denoise = { latent, timestep, context, _ in
        guard let target = context.conditioning["target"] else { return latent }
        return (latent - target * sqrtf(timestep.alphaBar)) / sqrtf(1 - timestep.alphaBar)
    }

    /// Nearest-neighbor upscale of an `[H, W, C]` tensor by an integer factor.
    static func nearestUpscale(_ image: MLXArray, scale: Int) -> MLXArray {
        let height = image.shape[0], width = image.shape[1], channels = image.shape[2]
        let expanded = image.reshaped([height, 1, width, 1, channels])
        let broadcasted = broadcast(expanded, to: [height, scale, width, scale, channels])
        return broadcasted.reshaped([height * scale, width * scale, channels])
    }
}
