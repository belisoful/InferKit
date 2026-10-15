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
        for release in wanAdditionalReleases {
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
        [ltxVideo] + wanEntries
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
        NFKMLXCatalogImage.model(NFKMLXLTXVideoGenerator.modelName, "LTX-Video 0.9.0", family: "ltx-video",
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
            return NFKMLXCatalogImage.model(release.name, release.displayName, family: "wan",
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
