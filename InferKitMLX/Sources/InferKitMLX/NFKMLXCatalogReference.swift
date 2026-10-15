//
//  NFKMLXCatalogReference.swift
//  InferKitMLX
//
//  The catalog entries of the reference stand-ins ``NFKMLXReferenceModels`` registers: the four oracle
//  diffusion pipelines, the green-screen keyer, and the tone synthesizer.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The catalog entries of the reference stand-ins.
///
/// @discussion Each stand-in is InferKit code built without weights, so its entry downloads nothing,
/// carries InferKit's own license, and is marked `standIn`. Its ports and options are the keys the
/// registered backend declares.
enum NFKMLXCatalogReference {

    /// The stand-ins' entries.
    static var entries: [NFKMLXCatalogEntry] {
        [diffusionControlNet, diffusionDepth, diffusionInpaint, diffusionUpscaler, greenScreenKeyer, toneSpeech]
    }

    /// The step count the diffusion stand-ins sample with where a request names none.
    static let diffusionSteps = 8

    /// The entry of the stand-in registered under `name`, with one variant that downloads nothing.
    static func standIn(_ name: String, displayName: String, family: String, summary: String, tasks: [String],
                        inputs: [NFKModelPort], outputs: [NFKModelPort],
                        parameters: [NFKModelOption]) -> NFKMLXCatalogEntry {
        NFKMLXCatalogEntry(identifier: name, displayName: displayName, family: family, summary: summary,
                           registryName: name, tasks: tasks, inputs: inputs, outputs: outputs,
                           parameters: parameters,
                           variants: [.variant("default", displayName: displayName, isDefault: true,
                                               registryName: name, source: .noDownload)],
                           license: .inferKit(note: "A reference stand-in built without weights."),
                           standIn: true)
    }

    /// The sampling options every ``NFKMLXDiffusionBackend`` reads.
    static var diffusionParameters: [NFKModelOption] {
        [
            .option(NFKParameterSteps, kind: NFKModelOptionKindInteger, title: "Steps",
                    help: "The denoising steps the sampler runs.",
                    defaultValue: diffusionSteps, minimum: 1, step: 1),
            .option(NFKParameterGuidanceScale, kind: NFKModelOptionKindNumber, title: "Guidance",
                    help: "The guidance weight the denoiser receives. The stand-in's oracle denoiser ignores it.",
                    defaultValue: 1, minimum: 0),
            .option(NFKParameterStrength, kind: NFKModelOptionKindNumber, title: "Strength",
                    help: "How much of a source image the sampler regenerates. 1 starts from full noise.",
                    defaultValue: 1, minimum: 0, maximum: 1),
            .option(NFKParameterSeed, kind: NFKModelOptionKindInteger, title: "Seed",
                    help: "The seed of the initial noise. A run without one uses a fixed seed."),
        ]
    }

    /// The input the diffusion loop bridges and a stand-in's encoder does not read.
    static func unusedDiffusionInput(_ key: String) -> NFKModelPort {
        .input(key, help: "The sampler bridges it, and this stand-in does not use it.")
    }

    static var diffusionControlNet: NFKMLXCatalogEntry {
        standIn("diffusion-controlnet", displayName: "Diffusion ControlNet (reference)", family: "reference-diffusion",
                summary: "A ControlNet-style pipeline whose oracle denoiser steers the output to the control map.",
                tasks: [NFKModelTaskControlNet],
                inputs: [.input(NFKInputControl, required: true, title: "Control",
                                help: "The control map the output follows: edges, depth, or pose."),
                         unusedDiffusionInput(NFKInputImage), unusedDiffusionInput(NFKInputMask)],
                outputs: [.output(NFKOutputImage, required: true, title: "Image",
                                  help: "The generated image, at the control map's size.")],
                parameters: diffusionParameters)
    }

    static var diffusionDepth: NFKMLXCatalogEntry {
        standIn("diffusion-depth", displayName: "Diffusion Depth (reference)", family: "reference-diffusion",
                summary: "A Marigold-style depth estimator whose oracle denoiser converges on the image's luminance.",
                tasks: [NFKModelTaskDepthEstimation],
                inputs: [.input(NFKInputImage, required: true, title: "Image", help: "The image to estimate depth for."),
                         unusedDiffusionInput(NFKInputMask)],
                outputs: [.output(NFKOutputImage, required: true, title: "Depth",
                                  help: "The depth map as a gray image at the input's size.")],
                parameters: diffusionParameters)
    }

    static var diffusionInpaint: NFKMLXCatalogEntry {
        standIn("diffusion-inpaint", displayName: "Diffusion Inpainting (reference)", family: "reference-diffusion",
                summary: "An inpainter that fills the masked region with flat gray through the full sampler.",
                tasks: [NFKModelTaskInpainting],
                inputs: [.input(NFKInputImage, required: true, title: "Image", help: "The plate to fill."),
                         .input(NFKInputMask, required: true, title: "Mask",
                                help: "White regenerates the region, and black keeps it.")],
                outputs: [.output(NFKOutputImage, required: true, title: "Image",
                                  help: "The plate with the masked region regenerated.")],
                parameters: diffusionParameters)
    }

    static var diffusionUpscaler: NFKMLXCatalogEntry {
        standIn("diffusion-upscaler", displayName: "Diffusion Upscaler (reference)", family: "reference-diffusion",
                summary: "A 2x upscaler whose oracle denoiser converges on a nearest-neighbor enlargement.",
                tasks: [NFKModelTaskUpscaling],
                inputs: [.input(NFKInputImage, required: true, title: "Image", help: "The image to enlarge."),
                         unusedDiffusionInput(NFKInputMask)],
                outputs: [.output(NFKOutputImage, required: true, title: "Image",
                                  help: "The image at twice the input's width and height.")],
                parameters: diffusionParameters)
    }

    static var greenScreenKeyer: NFKMLXCatalogEntry {
        standIn("green-screen-keyer", displayName: "Green-Screen Keyer (reference)", family: "reference-keyer",
                summary: "A keyer that derives an alpha matte from how strongly green dominates each pixel.",
                tasks: [NFKModelTaskMatting],
                inputs: [.input(NFKInputImage, required: true, title: "Plate", help: "The green-screen plate."),
                         .input(NFKInputMask, title: "Hint", help: "A hint mask. The stand-in keyer does not read it.")],
                outputs: [.output(NFKOutputImage, required: true, title: "Foreground",
                                  help: "The plate as straight foreground with the matte as alpha."),
                          .output(NFKOutputMask, required: true, title: "Matte",
                                  help: "The alpha matte on its own, as a gray image.")],
                parameters: [])
    }

    static var toneSpeech: NFKMLXCatalogEntry {
        standIn("tone-speech", displayName: "Tone Speech (reference)", family: "reference-tone-speech",
                summary: "A synthesizer that speaks each character as a tone whose pitch follows the character.",
                tasks: [NFKModelTaskSpeechSynthesis],
                inputs: [.input(NFKInputPrompt, title: "Text", help: "The text to speak."),
                         .input(NFKInputMessages, title: "Conversation",
                                help: "A conversation whose user turns are spoken, joined.")],
                outputs: [.output(NFKOutputAudio, required: true, title: "Audio", help: "A mono WAV file.")],
                parameters: [.option(NFKParameterSampleRate, kind: NFKModelOptionKindInteger, title: "Sample rate",
                                     help: "The output sample rate in hertz.",
                                     defaultValue: NFKMLXSpeechConfiguration().sampleRate, minimum: 1, step: 1)])
    }
}
