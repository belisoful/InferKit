//
//  NFKMLXCatalogVideo.swift
//  InferKitMLX
//
//  The catalog entries of the video models: video generation, frame interpolation, optical flow, video
//  super-resolution, video matting, video segmentation, video tokenization, and video understanding.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The catalog entries of the video models, and the registrations their registry names need.
///
/// @discussion The entries are built with ``NFKMLXCatalogImage``'s helpers, so an image model and its
/// clip backend describe their sources and licenses alike.
enum NFKMLXCatalogVideo {

    /// The video models' entries, in the order ``NFKMLXModelDescriptors`` lists them.
    static var entries: [NFKMLXCatalogEntry] {
        interpolators + [raft] + superResolvers + rvmClipEntries + sam2VideoEntries + generators
            + cosmosTokenizerEntries + vJEPA2Entries
    }

    /// Registers the video models that have a factory and no registry name, under the names their entries
    /// give. ``NFKMLXReferenceModels/registerAll()`` calls it.
    static func registerAdditionalModels() {
        for release in NFKMLXCatalogImage.sam2AdditionalReleases {
            NFKMLXModelRegistry.register(name: sam2VideoName(release)) { weightsURL in
                try NFKMLXSAM2.videoBackend(variant: release.variant, release: release.release, weightsURL: weightsURL)
            }
        }
        for release in wanAdditionalReleases + wanExpertReleases {
            NFKMLXGenerationBackend.registerDirectoryModel(release.name) {
                try NFKMLXWanVideoGenerator.backend(directoryURL: $0)
            }
        }
        for release in vJEPA2Releases {
            NFKMLXGenerationBackend.registerDirectoryModel(release.name) { try NFKMLXVJEPA2.backend(directoryURL: $0) }
        }
    }

    // MARK: - Ports

    /// The first of two frames a two-frame model reads.
    static var firstFrameInput: NFKModelPort {
        .input(NFKMLXRIFE.frame0Key, required: true, valueType: NFKModelValueTypeImage, title: "First frame",
               help: "The earlier frame.")
    }

    /// The second of two frames a two-frame model reads.
    static var secondFrameInput: NFKModelPort {
        .input(NFKMLXRIFE.frame1Key, required: true, valueType: NFKModelValueTypeImage, title: "Second frame",
               help: "The later frame, at the first frame's size.")
    }

    /// The clip an `NFKMLXVideoBackend` reads.
    static func clipInput(_ help: String) -> NFKModelPort {
        .input(NFKInputVideo, required: true, title: "Clip", help: help)
    }

    /// The clip a video backend writes.
    static func clipOutput(_ help: String) -> NFKModelPort {
        .output(NFKOutputVideo, required: true, title: "Clip", help: help)
    }

    /// The frame rate an `NFKMLXVideoBackend` writes its clip at.
    static func frameRateOption(_ help: String) -> NFKModelOption {
        .option(NFKParameterFramesPerSecond, title: "Frame rate", help: help, minimum: 1)
    }

    // MARK: - Frame interpolation

    static let rifeLicense = NFKModelLicense.mit(url: "https://github.com/hzwer/ECCV2022-RIFE/blob/main/LICENSE",
                                                 note: "Served from a Hugging Face mirror of the authors' release.")

    static let rifeSource = NFKMLXCatalogImage.hubFile("yow46228/RIFE", "flownet.pkl", requiresConversion: true)

    static var interpolators: [NFKMLXCatalogEntry] {
        [
            NFKMLXCatalogImage.model(NFKMLXRIFE.modelName, "RIFE HDv3", family: "rife",
                                     summary: "A frame interpolator that synthesizes the frame midway between two frames.",
                                     tasks: [NFKModelTaskFrameInterpolation],
                                     inputs: [firstFrameInput, secondFrameInput],
                                     outputs: [NFKMLXCatalogImage.imageOutput("The frame midway between the two.",
                                                                              title: "Middle frame")],
                                     license: rifeLicense, source: rifeSource),
            NFKMLXCatalogImage.model(NFKMLXRIFE.clipModelName, "RIFE HDv3 (clip)", family: "rife",
                                     summary: "A frame interpolator that doubles a clip's frame rate.",
                                     tasks: [NFKModelTaskFrameInterpolation],
                                     inputs: [clipInput("The clip to interpolate.")],
                                     outputs: [clipOutput("The clip with a synthesized frame between each adjacent pair: n frames become 2n − 1.")],
                                     parameters: [frameRateOption("The frame rate the clip is written at. Twice the source's when absent.")],
                                     license: rifeLicense, source: rifeSource),
            NFKMLXCatalogImage.model(NFKMLXRIFEv4.modelName, "RIFE v4.13", family: "rife",
                                     summary: "A frame interpolator that synthesizes the frame at any point between two frames.",
                                     tasks: [NFKModelTaskFrameInterpolation],
                                     inputs: [firstFrameInput, secondFrameInput],
                                     outputs: [NFKMLXCatalogImage.imageOutput("The frame at the timestep between the two.",
                                                                              title: "Interpolated frame")],
                                     parameters: [.option(NFKMLXRIFEv4.timestepKey, kind: NFKModelOptionKindNumber,
                                                          title: "Timestep",
                                                          help: "Where the frame lands between the first frame (0) and the second (1).",
                                                          defaultValue: 0.5, minimum: 0, maximum: 1)],
                                     license: .mit(url: "https://github.com/hzwer/Practical-RIFE/blob/main/LICENSE"),
                                     source: NFKMLXCatalogImage.releasePage("https://github.com/hzwer/Practical-RIFE",
                                                                            requiresConversion: true)),
        ]
    }

    // MARK: - Optical flow

    static var raft: NFKMLXCatalogEntry {
        NFKMLXCatalogImage.model(NFKMLXRAFT.modelName, "RAFT (Things)", family: "raft",
                                 summary: "An optical-flow estimator of each pixel's motion between two frames.",
                                 tasks: [NFKModelTaskOpticalFlow],
                                 inputs: [firstFrameInput, secondFrameInput],
                                 outputs: [NFKMLXCatalogImage.imageOutput("The motion from the first frame to the second as a packed flow map.",
                                                                          title: "Flow")],
                                 license: .license("BSD-3-Clause", name: "BSD 3-Clause License",
                                                   url: "https://github.com/princeton-vl/RAFT/blob/master/LICENSE"),
                                 source: NFKMLXCatalogImage.download("https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip",
                                                                     requiresConversion: true))
    }

    // MARK: - Video super-resolution and matting

    static let basicVSRSource = NFKMLXCatalogImage.download(
        "https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth",
        requiresConversion: true)

    static let basicVSRLicense = NFKModelLicense.license("Apache-2.0", name: "Apache License 2.0",
                                                         url: "https://github.com/open-mmlab/mmagic/blob/main/LICENSE")

    static var superResolvers: [NFKMLXCatalogEntry] {
        [
            NFKMLXCatalogImage.model(NFKMLXVideoSR.modelName, "BasicVSR (REDS4, one frame)", family: "basicvsr",
                                     summary: "A video upscaler that enlarges one frame ×4 without its neighbors.",
                                     tasks: [NFKModelTaskUpscaling],
                                     inputs: [NFKMLXCatalogImage.imageInput("The frame to enlarge.", title: "Frame")],
                                     outputs: [NFKMLXCatalogImage.imageOutput("The frame at four times its width and height.",
                                                                              title: "Frame")],
                                     license: basicVSRLicense, source: basicVSRSource),
            NFKMLXCatalogImage.model(NFKMLXVideoSR.clipModelName, "BasicVSR (REDS4)", family: "basicvsr",
                                     summary: "A video upscaler that enlarges a clip ×4, each frame drawing on the frames around it.",
                                     tasks: [NFKModelTaskUpscaling],
                                     inputs: [clipInput("The clip to enlarge.")],
                                     outputs: [clipOutput("The clip at four times its width and height.")],
                                     parameters: [frameRateOption("The frame rate the clip is written at. The source's when absent.")],
                                     license: basicVSRLicense, source: basicVSRSource),
        ]
    }

    static var rvmClipEntries: [NFKMLXCatalogEntry] {
        NFKMLXCatalogImage.rvmReleases.map { variant in
            let release = NFKMLXCatalogImage.rvmRelease(variant)
            return NFKMLXCatalogImage.model(NFKMLXRVM.clipModelName(variant: variant),
                                            "Robust Video Matting \(release.encoder)", family: "robust-video-matting",
                                            summary: "A human matter that mattes a clip, each frame drawing on the frames before it.",
                                            tasks: [NFKModelTaskMatting],
                                            inputs: [clipInput("The clip to matte.")],
                                            outputs: [clipOutput("The alpha matte of every frame as a clip.")],
                                            parameters: [frameRateOption("The frame rate the clip is written at. The source's when absent.")],
                                            license: NFKMLXCatalogImage.rvmLicense,
                                            source: NFKMLXCatalogImage.download(release.address, requiresConversion: true))
        }
    }

    // MARK: - Video segmentation

    /// The video backend's registry name for a SAM 2 size and release.
    static func sam2VideoName(_ release: NFKMLXCatalogImage.SAM2Release) -> String {
        NFKMLXCatalogImage.sam2Name(release) + "-video"
    }

    static var sam2VideoEntries: [NFKMLXCatalogEntry] {
        let defaultRelease = NFKMLXCatalogImage.sam2DefaultRelease
        return ([defaultRelease] + NFKMLXCatalogImage.sam2AdditionalReleases).map { release in
            let isDefault = release.variant == defaultRelease.variant && release.release == defaultRelease.release
            return NFKMLXCatalogImage.model(isDefault ? NFKMLXSAM2.videoModelName : sam2VideoName(release),
                                            NFKMLXCatalogImage.sam2DisplayName(release) + " (video)", family: "sam2",
                                            summary: "A promptable segmenter that tracks an object through a clip from a prompt on its first frame.",
                                            tasks: [NFKModelTaskSegmentation],
                                            inputs: [clipInput("The clip to track the object through.")],
                                            outputs: [clipOutput("The object's mask on every frame as a clip."),
                                                      .output(NFKOutputStructured, required: true, title: "Scores",
                                                              help: "The tracker's object score on each frame.")],
                                            parameters: [.option(NFKSAMPointKey, kind: NFKModelOptionKindJSON, title: "Point",
                                                                 help: "A click on the object on the first frame, an array [x, y] in pixels."),
                                                         .option(NFKMLXSAM2ParameterKey.box, kind: NFKModelOptionKindJSON,
                                                                 title: "Box",
                                                                 help: "A box around the object on the first frame, an array [x0, y0, x1, y1] in pixels. With neither a point nor a box the frame's center is clicked.")],
                                            license: NFKMLXCatalogImage.apacheSAM2,
                                            source: NFKMLXCatalogImage.sam2Source(release))
        }
    }

    // MARK: - Video generation

    static var generators: [NFKMLXCatalogEntry] {
        [ltxVideo] + ltx2Entries + wanEntries + wanExpertEntries + wanAnimateEntries
    }

    /// The options a text-to-video generator reads.
    static func videoOptions(frames: Int, width: Int, height: Int, framesPerSecond: Double?, frameRateHelp: String,
                             steps: Int, guidance: Double) -> [NFKModelOption] {
        [.option(NFKParameterFrameCount, title: "Frames", help: "The frames generated.", defaultValue: frames,
                 minimum: 1, step: 1),
         NFKMLXCatalogImage.widthOption(width, help: "The clip width in pixels."),
         NFKMLXCatalogImage.heightOption(height, help: "The clip height in pixels."),
         .option(NFKParameterFramesPerSecond, title: "Frame rate", help: frameRateHelp, defaultValue: framesPerSecond,
                 minimum: 1),
         NFKMLXCatalogImage.seedOption(0), NFKMLXCatalogImage.stepsOption(steps),
         NFKMLXCatalogImage.guidanceOption(guidance,
                                           help: "The classifier-free guidance weight. Above 1 the clip guides against the negative prompt.")]
    }

    static var promptInputs: [NFKModelPort] {
        [.input(NFKInputPrompt, required: true, title: "Prompt", help: "What to film."),
         .input(NFKInputNegativePrompt, title: "Negative prompt", help: "What to steer away from.")]
    }

    static var ltxVideo: NFKMLXCatalogEntry {
        var entry = NFKMLXCatalogImage.model(NFKMLXLTXVideoGenerator.modelName, "LTX-Video 0.9.0", family: "ltx-video",
                                             summary: "A diffusion transformer that films a clip from a prompt.",
                                             tasks: [NFKModelTaskVideoGeneration],
                                             inputs: promptInputs,
                                             outputs: [clipOutput("The generated clip as a QuickTime movie.")],
                                             parameters: videoOptions(frames: 121, width: 704, height: 480, framesPerSecond: 25,
                                                                      frameRateHelp: "The frame rate the clip is conditioned on and written at.",
                                                                      steps: 50, guidance: 3),
                                             license: .license(nil, name: "LTX Video 0.9 RAIL-M License",
                                                               url: "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.license.txt",
                                                               note: "Use restrictions apply."),
                                             source: NFKMLXCatalogImage.hubRelease(NFKMLXLTXVideoGenerator.releaseRepo,
                                                                                   components: NFKMLXLTXVideoGenerator.releaseComponents),
                                             residencies: NFKMLXCatalogImage.stagedResidencies)
        entry.takesAdapters(trainedOn: [NFKMLXLTXVideoGenerator.releaseRepo])
        return entry
    }

    /// LTX-2.5's prompt enhancement, which reads the release's `prompt_enhancer/` and `processor/` where its
    /// directory holds them.
    static var ltx2PromptEnhancementOption: NFKModelOption {
        .option(NFKMLXLTX2ParameterKey.promptEnhancement, kind: NFKModelOptionKindBoolean, title: "Enhance the prompt",
                help: "Rewrites the prompt as a long audio-visual caption with the release's Gemma 4 E2B enhancer "
                    + "before generating. Needs the release's prompt_enhancer/ and processor/ folders (about 10 GB).",
                defaultValue: false)
    }

    /// The guidance options LTX-2's sampler reads beside the video's classifier-free guidance.
    static var ltx2GuidanceOptions: [NFKModelOption] {
        [.option(NFKMLXLTX2ParameterKey.audioGuidanceScale, title: "Soundtrack guidance",
                 help: "The soundtrack's classifier-free guidance weight.", defaultValue: 7, minimum: 0),
         .option(NFKMLXLTX2ParameterKey.spatioTemporalGuidanceScale, title: "Spatio-temporal guidance",
                 help: "The weight of the pass whose self-attention is perturbed. 0 skips that pass.", defaultValue: 1,
                 minimum: 0),
         .option(NFKMLXLTX2ParameterKey.modalityGuidanceScale, title: "Modality guidance",
                 help: "The weight of the pass that hears no video and sees no audio. 1 skips that pass.",
                 defaultValue: 3, minimum: 0),
         .option(NFKMLXLTX2ParameterKey.guidanceRescale, title: "Guidance rescale",
                 help: "How far the guided prediction is pulled back to the conditional one's spread. 0 turns it off.",
                 defaultValue: 0.7, minimum: 0, maximum: 1)]
    }

    /// The LTX-2.5 decode-side switches: the diffusion decoder, the two-stage distilled recipe, and the
    /// temporal latent upsampler, each off by default.
    static var ltx25DecodePathOptions: [NFKModelOption] {
        [.option(NFKMLXLTX2ParameterKey.diffusionDecode, kind: NFKModelOptionKindBoolean, title: "Diffusion decode",
                 help: "Decode the clip with the release's diffusion decoder instead of the autoencoder.",
                 defaultValue: false),
         .option(NFKMLXLTX2ParameterKey.twoStageRefinement, kind: NFKModelOptionKindBoolean, title: "Two-stage refinement",
                 help: "Sample at half the size, upsample the latent, and refine it over three sigmas at the full size.",
                 defaultValue: false),
         .option(NFKMLXLTX2ParameterKey.temporalUpsampling, kind: NFKModelOptionKindBoolean, title: "Temporal upsampling",
                 help: "Double the frame count and the frame rate with the release's temporal latent upsampler.",
                 defaultValue: false)]
    }

    /// LTX-2.5's Diffusion Fidelity Rendering switches, off by default.
    static var ltx25DFROptions: [NFKModelOption] {
        [.option(NFKMLXLTX2ParameterKey.dfr, kind: NFKModelOptionKindBoolean, title: "Diffusion Fidelity Rendering",
                 help: "Render by the DFR recipe: keyframe slots at half size, a full-size pass with the detailing "
                     + "IC-LoRA where it is loaded, temporal refine rounds, and an optional third spatial stage.",
                 defaultValue: false),
         .option(NFKMLXLTX2ParameterKey.dfrTemporalRounds, kind: NFKModelOptionKindInteger, title: "DFR temporal rounds",
                 help: "Each round doubles the frame count and the frame rate.", defaultValue: 1, minimum: 0, maximum: 2,
                 step: 1),
         .option(NFKMLXLTX2ParameterKey.dfrEpilogue, kind: NFKModelOptionKindBoolean, title: "DFR epilogue",
                 help: "Refine the result again at twice the width and height.", defaultValue: false)]
    }

    /// The LTX-2 releases: one transformer that films a clip and its soundtrack together.
    static var ltx2Entries: [NFKMLXCatalogEntry] {
        NFKMLXLTX2Variant.allCases.map { variant in
            let license: NFKModelLicense
            switch variant {
            case .ltx23:
                license = .license(nil, name: "LTX-2 Community License Agreement",
                                   url: "https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2",
                                   restrictsCommercialUse: true,
                                   note: "Entities with annual revenue of US$10 million or more need a paid commercial license.")
            case .ltx25, .ltx25Full:
                license = .license(nil, name: "LTX-2.x Community License Agreement",
                                   url: "https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x", gated: true,
                                   restrictsCommercialUse: true,
                                   note: "Entities with annual revenue of US$10 million or more need a paid commercial license.")
            }
            return NFKMLXCatalogImage.model(
                variant.modelName, variant.displayName, family: variant == .ltx25Full ? "ltx-2-full" : "ltx-2",
                summary: "A diffusion transformer that films a clip and its soundtrack together from a prompt.",
                tasks: [NFKModelTaskVideoGeneration],
                inputs: promptInputs + (variant == .ltx25
                    ? [.input(NFKInputImage, title: "First frame",
                              help: "A still the DFR recipe opens on, encoded at each stage's size.")] : []),
                outputs: [clipOutput("The generated clip as a QuickTime movie whose audio track is the soundtrack."),
                          .output(NFKOutputAudio, required: true, title: "Soundtrack",
                                  help: "The generated soundtrack alone, a stereo WAV at 48 kHz.")],
                parameters: videoOptions(frames: 121, width: 768, height: 512, framesPerSecond: 24,
                                         frameRateHelp: "The frame rate the clip is conditioned on and written at.",
                                         steps: 30, guidance: 3) + ltx2GuidanceOptions
                    + (variant == .ltx23 ? [] : ltx25DecodePathOptions + [ltx2PromptEnhancementOption])
                    + (variant == .ltx25 ? ltx25DFROptions : []),
                license: license,
                source: NFKMLXCatalogImage.hubRelease(variant.releaseRepo,
                                                      components: NFKMLXLTX2Generator.releaseComponents(for: variant)),
                residencies: NFKMLXCatalogImage.streamedStageResidencies)
        }
    }

    /// The Wan releases other than Wan 2.1 T2V 1.3B, which `wan` builds.
    static let wanAdditionalReleases: [NFKMLXCatalogImage.GeneratorRelease] = [
        NFKMLXCatalogImage.GeneratorRelease(name: "wan2.1-t2v-14b", repository: "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                                            displayName: "Wan 2.1 T2V 14B", license: .apache2()),
        NFKMLXCatalogImage.GeneratorRelease(name: "wan2.2-ti2v-5b", repository: "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                                            displayName: "Wan 2.2 TI2V 5B", license: .apache2()),
    ]

    static var wanEntries: [NFKMLXCatalogEntry] {
        let releases = [NFKMLXCatalogImage.GeneratorRelease(name: NFKMLXWanVideoGenerator.modelName,
                                                            repository: NFKMLXWanVideoGenerator.releaseRepo,
                                                            displayName: "Wan 2.1 T2V 1.3B", license: .apache2())]
            + wanAdditionalReleases
        return releases.map { release in
            let framesPerSecond: Double = release.name == "wan2.2-ti2v-5b" ? 24 : 16
            var entry = NFKMLXCatalogImage.model(release.name, release.displayName, family: "wan",
                                            summary: "A diffusion transformer that films a clip from a prompt.",
                                            tasks: [NFKModelTaskVideoGeneration],
                                            inputs: promptInputs,
                                            outputs: [clipOutput("The generated clip as a QuickTime movie.")],
                                            parameters: videoOptions(frames: 81, width: 832, height: 480,
                                                                     framesPerSecond: framesPerSecond,
                                                                     frameRateHelp: "The frame rate the clip is written at, the release's own when absent.",
                                                                     steps: 50, guidance: 5),
                                            license: release.license,
                                            source: NFKMLXCatalogImage.hubRelease(release.repository,
                                                                                  components: NFKMLXWanVideoGenerator.releaseComponents),
                                            residencies: NFKMLXCatalogImage.stagedResidencies)
            // Wan 2.1 stores its transformer at float32, and testWan21TransformerInBFloat16MatchesTheBFloat16Reference
            // holds its bfloat16 load to diffusers' bfloat16 run.
            if release.name != "wan2.2-ti2v-5b" {
                entry.load = .diffusionPipeline(widenedStages: ["text_encoder": NFKModelPrecisionFloat32],
                                                precisions: [NFKModelPrecisionCheckpoint, NFKModelPrecisionBFloat16],
                                                precisionStage: "transformer")
            }
            return entry
        }
    }

    /// The Wan 2.2 A14B releases, each two 14B experts that split the schedule at the release's boundary.
    static let wanExpertReleases: [NFKMLXCatalogImage.GeneratorRelease] = [
        NFKMLXCatalogImage.GeneratorRelease(name: "wan2.2-t2v-a14b", repository: NFKMLXWanVideoGenerator.textToVideoExpertsRepo,
                                            displayName: "Wan 2.2 T2V A14B", license: .apache2()),
        NFKMLXCatalogImage.GeneratorRelease(name: "wan2.2-i2v-a14b", repository: NFKMLXWanVideoGenerator.imageToVideoExpertsRepo,
                                            displayName: "Wan 2.2 I2V A14B", license: .apache2()),
    ]

    static var wanExpertEntries: [NFKMLXCatalogEntry] {
        wanExpertReleases.map { release in
            let imageToVideo = release.name == "wan2.2-i2v-a14b"
            let inputs = imageToVideo
                ? [NFKMLXCatalogImage.imageInput("The clip's first frame, resized to the clip's size.", title: "First frame"),
                   .input(NFKMLXVideoInputKey.lastImage, title: "Last frame",
                          help: "The clip's last frame, resized to the clip's size.")]
                    + promptInputs
                : promptInputs
            return NFKMLXCatalogImage.model(
                release.name, release.displayName, family: "wan-a14b",
                summary: imageToVideo
                    ? "Two diffusion transformers that film a clip from a first frame and a prompt, one for the noisy steps and one for the rest."
                    : "Two diffusion transformers that film a clip from a prompt, one for the noisy steps and one for the rest.",
                tasks: [NFKModelTaskVideoGeneration],
                inputs: inputs,
                outputs: [clipOutput("The generated clip as a QuickTime movie.")],
                parameters: videoOptions(frames: 81, width: 832, height: 480, framesPerSecond: 16,
                                         frameRateHelp: "The frame rate the clip is written at.", steps: 50, guidance: 5)
                    + [.option(NFKMLXGenerationParameterKey.lowNoiseGuidanceScale, title: "Low-noise guidance",
                               help: "The guidance weight of the expert that runs the later steps; the guidance weight when absent.",
                               defaultValue: nil, minimum: 0)],
                license: release.license,
                source: NFKMLXCatalogImage.hubRelease(release.repository,
                                                      components: NFKMLXWanVideoGenerator.expertReleaseComponents),
                residencies: NFKMLXCatalogImage.streamedStageResidencies)
        }
    }

    /// Wan 2.2 Animate 2's two releases, ``NFKMLXWanAnimateGenerator/modelName`` and
    /// ``NFKMLXWanAnimateGenerator/distilledModelName``. ``NFKMLXWanAnimateGenerator/register()`` registers both.
    static var wanAnimateEntries: [NFKMLXCatalogEntry] {
        [(NFKMLXWanAnimateVariant.standard, NFKMLXWanAnimateGenerator.modelName, "Wan 2.2 Animate 2 14B"),
         (.distilled, NFKMLXWanAnimateGenerator.distilledModelName, "Wan 2.2 Animate 2 14B Distilled")].map {
            variant, name, displayName in
            NFKMLXCatalogImage.model(
                name, displayName, family: "wan-animate",
                summary: "A diffusion transformer that animates the character in a reference image with the motion of a driving video.",
                tasks: [NFKModelTaskVideoGeneration],
                inputs: [NFKMLXCatalogImage.imageInput("The character to animate.", title: "Reference image"),
                         .input(NFKInputVideo, required: true, title: "Driving video",
                                help: "The motion to follow, resampled to the frame rate."),
                         .input(NFKInputPrompt, title: "Prompt", help: "A description of the character's appearance and the background."),
                         .input(NFKInputNegativePrompt, title: "Negative prompt",
                                help: "What to steer away from, read where the guidance is not 1.")],
                outputs: [clipOutput("The animated clip as a QuickTime movie, one frame per resampled driving frame.")],
                parameters: [
                    NFKMLXCatalogImage.widthOption(640, help: "With the height, the frame's area; the reference image sets its shape."),
                    NFKMLXCatalogImage.heightOption(800, help: "With the width, the frame's area."),
                    .option(NFKParameterFramesPerSecond, title: "Frame rate",
                            help: "The rate the driving video is resampled to and the clip plays at.", defaultValue: 24.0,
                            minimum: 1),
                    NFKMLXCatalogImage.seedOption(0), NFKMLXCatalogImage.stepsOption(variant.defaultSteps),
                    NFKMLXCatalogImage.guidanceOption(Double(variant.defaultGuidance),
                                                      help: "The classifier-free guidance weight. Other than 1 the clip guides against the negative prompt."),
                    .option(NFKMLXGenerationParameterKey.segmentLength, title: "Segment length",
                            help: "The frames each segment denoises.", defaultValue: 81, minimum: 2, step: 1),
                    .option(NFKMLXGenerationParameterKey.carriedFrames, title: "Carried frames",
                            help: "The frames each segment carries over from the one before it.", defaultValue: 1,
                            minimum: 0, step: 1),
                ],
                license: .apache2(),
                source: NFKMLXCatalogImage.hubRelease(variant.repository, components: NFKMLXWanAnimateGenerator.releaseComponents),
                residencies: NFKMLXCatalogImage.streamedStageResidencies)
        }
    }

    // MARK: - Video tokenization

    static var cosmosTokenizerEntries: [NFKMLXCatalogEntry] {
        NFKMLXCosmosTokenizerVariant.allCases.map { variant in
            let kind = (variant.isDiscrete ? "discrete" : "continuous") + (variant.isVideo ? " video" : " image")
            let inputs: [NFKModelPort]
            let outputs: [NFKModelPort]
            if variant.isVideo {
                inputs = [.input(NFKInputVideo, title: "Clip", help: "The clip to encode and decode."),
                          .input(NFKInputImage, title: "Image", help: "An image to encode and decode as a one-frame clip.")]
                outputs = [.output(NFKOutputVideo, title: "Clip", help: "The clip decoded from its tokens."),
                           .output(NFKOutputImage, title: "Image", help: "The image decoded from its tokens.")]
            } else {
                inputs = [NFKMLXCatalogImage.imageInput("The image to encode and decode.")]
                outputs = [NFKMLXCatalogImage.imageOutput("The image decoded from its tokens.")]
            }
            return NFKMLXCatalogImage.model(variant.modelName, "Cosmos Tokenizer \(variant.releaseName)",
                                            family: "cosmos-tokenizer",
                                            summary: "A \(kind) tokenizer that compresses its input to latent tokens and reconstructs it.",
                                            tasks: [NFKModelTaskImageToImage],
                                            inputs: inputs, outputs: outputs,
                                            license: .license(nil, name: "NVIDIA Open Model License",
                                                              url: "https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf"),
                                            source: NFKMLXCatalogImage.hubFile(variant.repository, "autoencoder.jit",
                                                                               requiresConversion: false))
        }
    }

    // MARK: - Video understanding

    /// The V-JEPA 2 releases: four encoders and four video classifiers.
    static let vJEPA2Releases: [NFKMLXCatalogImage.DirectoryRelease] = [
        ("vitl-fpc64-256", "ViT-L, 64 frames, 256 px"), ("vith-fpc64-256", "ViT-H, 64 frames, 256 px"),
        ("vitg-fpc64-256", "ViT-g, 64 frames, 256 px"), ("vitg-fpc64-384", "ViT-g, 64 frames, 384 px"),
        ("vitl-fpc16-256-ssv2", "ViT-L, Something-Something v2 classifier"),
        ("vitl-fpc32-256-diving48", "ViT-L, Diving48 classifier"),
        ("vitg-fpc64-384-ssv2", "ViT-g, Something-Something v2 classifier"),
        ("vitg-fpc32-384-diving48", "ViT-g, Diving48 classifier"),
    ].map { stem, description in
        NFKMLXCatalogImage.DirectoryRelease(name: "vjepa2-\(stem)", repository: "facebook/vjepa2-\(stem)",
                                            displayName: "V-JEPA 2 \(description)")
    }

    static var vJEPA2Entries: [NFKMLXCatalogEntry] {
        vJEPA2Releases.map { release in
            let classifies = release.name.hasSuffix("ssv2") || release.name.hasSuffix("diving48")
            let summary = classifies
                ? "A self-supervised video transformer with an attentive classifier that names the action in a clip."
                : "A self-supervised video transformer whose pooled features describe a clip or an image."
            return NFKMLXCatalogImage.model(release.name, release.displayName, family: "vjepa2", summary: summary,
                                            tasks: classifies ? [NFKModelTaskClassification, NFKModelTaskImageEmbedding]
                                                              : [NFKModelTaskImageEmbedding],
                                            inputs: [.input(NFKInputVideo, title: "Clip", help: "The clip to describe."),
                                                     .input(NFKInputImage, title: "Image",
                                                            help: "An image to describe as a still clip. Read when no clip is given.")],
                                            outputs: [.output(NFKOutputEmbedding, required: true, title: "Features",
                                                              help: "The mean-pooled feature embedding."),
                                                      .output(NFKOutputClassifications, title: "Classes",
                                                              help: "The ranked classes of a classifier release.")],
                                            license: .mit(url: "https://huggingface.co/\(release.repository)"),
                                            source: NFKMLXCatalogImage.hubRelease(release.repository,
                                                                                  required: NFKMLXVJEPA2.requiredFiles,
                                                                                  optional: NFKMLXVJEPA2.optionalFiles,
                                                                                  weights: NFKMLXVJEPA2.weightFiles))
        }
    }
}
