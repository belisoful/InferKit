//
//  NFKMLXCatalogLive.swift
//  InferKitMLX
//
//  The catalog's models that run on a live source, one video frame or audio chunk at a time.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The live profiles of the catalog's models (`NFKModelDescriptor.liveProfile`).
///
/// @discussion A model is declared live when it is built to answer one frame in a bounded single
/// forward: a small segmenter, matter, detector, depth or pose estimator, or upscaler. Each declares the
/// frame's input key and every output key it writes. A stateful model carries state from frame to frame
/// through the live stream its backend opens (`openLiveStream(parameters:)`).
enum NFKMLXCatalogLive {

    /// The entry with its live profile and the profile's measured timings, where it has one.
    static func live(_ entry: NFKMLXCatalogEntry) -> NFKMLXCatalogEntry {
        guard let profile = profile(entry.identifier) else {
            return entry
        }
        var live = entry
        live.liveProfile = timed(profile, timings[entry.identifier] ?? [])
        return live
    }

    private static func profile(_ identifier: String) -> NFKModelLiveProfile? {
        if let audio = audioChunks[identifier] {
            return audio
        }
        let stateful = statefulVideoFrameOutputs[identifier] != nil
        guard let outputs = videoFrameOutputs[identifier] ?? statefulVideoFrameOutputs[identifier] else {
            return nil
        }
        return NFKModelLiveProfile.videoFrameProfile(withInputKeys: [NFKInputImage], outputKeys: outputs,
                                                     stateful: stateful, timings: nil)
    }

    /// `profile` carrying `timings`.
    static func timed(_ profile: NFKModelLiveProfile, _ timings: [NFKModelLiveTiming]) -> NFKModelLiveProfile {
        guard !timings.isEmpty else {
            return profile
        }
        return NFKModelLiveProfile(unit: profile.unit, inputKeys: profile.inputKeys, outputKeys: profile.outputKeys,
                                   stateful: profile.isStateful, chunkDuration: profile.chunkDuration,
                                   sampleRate: profile.sampleRate, channelCount: profile.channelCount,
                                   lookaheadMilliseconds: profile.lookaheadMilliseconds, timings: timings,
                                   extensions: profile.extensions)
    }

    /// The measured time per unit of each live model through the stream its backend opens, by catalog
    /// identifier, as `NFKMLXLiveTimingTests` reports it in a release build: a video model at 1280 × 720
    /// and 1920 × 1080, an audio model per profile chunk. Each time includes the unit's conversion to and
    /// from the model's tensors.
    static let timings: [String: [NFKModelLiveTiming]] = [
        "bisenet": [video(1280, 720, 86.46), video(1920, 1080, 185.98)],
        "bisenet-v2": [video(1280, 720, 79.51), video(1920, 1080, 177.35)],
        "deepfilternet3": [audio(6.320)],
        "deeplabv3": [video(1280, 720, 280.11), video(1920, 1080, 625.15)],
        "depth-anything-3-small": [video(1280, 720, 191.70), video(1920, 1080, 204.34)],
        "depth-anything-v2-small": [video(1280, 720, 60.18), video(1920, 1080, 74.06)],
        "frcrn": [audio(48.98)],
        "gtcrn": [audio(15.26)],
        "modnet": [video(1280, 720, 137.20), video(1920, 1080, 302.63)],
        "nemotron-asr-streaming": [audio(30.96)],
        "pose-simplebaseline": [video(1280, 720, 19.49), video(1920, 1080, 27.54)],
        "real-esrgan-general-x4v3": [video(1280, 720, 672.69), video(1920, 1080, 1497.9)],
        "retinaface-mobile025": [video(1280, 720, 17.29), video(1920, 1080, 32.55)],
        "rf-detr-nano": [video(1280, 720, 22.68), video(1920, 1080, 30.95)],
        "rf-detr-seg-small": [video(1280, 720, 29.11), video(1920, 1080, 37.24)],
        "rf-detr-small": [video(1280, 720, 29.46), video(1920, 1080, 37.67)],
        "robust-video-matting": [video(1280, 720, 52.45), video(1920, 1080, 81.91)],
        "robust-video-matting-resnet50": [video(1280, 720, 61.73), video(1920, 1080, 91.24)],
        "rtdetr-r18vd": [video(1280, 720, 36.18), video(1920, 1080, 44.51)],
        "segformer-b0": [video(1280, 720, 102.18), video(1920, 1080, 268.66)],
        "silero-vad": [audio(0.849)],
        "swinir-lightweight-x2": [
            video(1280, 720, 4100.9, note: "The median of 15 frames, as many as a minute of timing holds."),
            video(1920, 1080, 14884.9, note: "The median of 5 frames, as many as a minute of timing holds."),
        ],
        "swinir-lightweight-x3": [
            video(1280, 720, 4157.9, note: "The median of 15 frames, as many as a minute of timing holds."),
            video(1920, 1080, 15198.7, note: "The median of 4 frames, as many as a minute of timing holds."),
        ],
        "swinir-lightweight-x4": [
            video(1280, 720, 4181.4, note: "The median of 15 frames, as many as a minute of timing holds."),
            video(1920, 1080, 15999.8, note: "The median of 4 frames, as many as a minute of timing holds."),
        ],
        "u2netp": [video(1280, 720, 282.43), video(1920, 1080, 600.75)],
    ]

    /// The machine and the day of the measurements in ``timings``.
    static let measuredChip = "Apple M1 Max"
    static let measuredDate = "2026-11-11"

    private static func video(_ width: Int, _ height: Int, _ milliseconds: Double, note: String? = nil) -> NFKModelLiveTiming {
        NFKModelLiveTiming(variantIdentifier: nil, chip: measuredChip, width: width, height: height,
                           millisecondsPerUnit: milliseconds, measuredDate: measuredDate, note: note, extensions: nil)
    }

    private static func audio(_ milliseconds: Double, note: String? = nil) -> NFKModelLiveTiming {
        video(0, 0, milliseconds, note: note)
    }

    /// The audio-chunk profile of each audio model whose backend opens a live stream. A chunk is the
    /// model's own step: a hop for a frame-causal enhancer, a decoding stride for a windowed one. The
    /// lookahead is the most audio the stream reads past a sample before it answers that sample.
    static let audioChunks: [String: NFKModelLiveProfile] = [
        "silero-vad": .audioChunkProfile(withInputKeys: [NFKInputAudio], outputKeys: [NFKOutputSegments], stateful: true,
                                         chunkDuration: 0.032, sampleRate: 16000, channelCount: 1,
                                         lookaheadMilliseconds: 0, timings: nil),
        "gtcrn": .audioChunkProfile(withInputKeys: [NFKInputAudio], outputKeys: [NFKOutputAudio], stateful: true,
                                    chunkDuration: 0.016, sampleRate: 16000, channelCount: 1,
                                    lookaheadMilliseconds: 32, timings: nil),
        "deepfilternet3": .audioChunkProfile(withInputKeys: [NFKInputAudio], outputKeys: [NFKOutputAudio], stateful: true,
                                             chunkDuration: 0.010, sampleRate: 48000, channelCount: 1,
                                             lookaheadMilliseconds: 40, timings: nil),
        "nemotron-asr-streaming": .audioChunkProfile(withInputKeys: [NFKInputAudio], outputKeys: [NFKOutputText, NFKOutputSegments],
                                                     stateful: true, chunkDuration: 1.12, sampleRate: 16000, channelCount: 1,
                                                     lookaheadMilliseconds: 1139, timings: nil),
        "frcrn": .audioChunkProfile(withInputKeys: [NFKInputAudio], outputKeys: [NFKOutputAudio], stateful: true,
                                    chunkDuration: 0.75, sampleRate: 16000, channelCount: 1,
                                    lookaheadMilliseconds: 1000, timings: nil),
    ]

    /// The output keys of each video-frame model that carries state from frame to frame.
    static let statefulVideoFrameOutputs: [String: [String]] = [
        "robust-video-matting": [NFKOutputImage, NFKOutputMask],
        "robust-video-matting-resnet50": [NFKOutputImage, NFKOutputMask],
    ]

    /// The output keys of each stateless video-frame model, by catalog identifier.
    static let videoFrameOutputs: [String: [String]] = [
        "modnet": [NFKOutputImage, NFKOutputMask],
        "u2netp": [NFKOutputImage, NFKOutputMask],
        "bisenet": [NFKOutputImage],
        "bisenet-v2": [NFKOutputImage],
        "segformer-b0": [NFKOutputImage],
        "deeplabv3": [NFKOutputImage],
        "depth-anything-v2-small": [NFKOutputImage],
        "depth-anything-3-small": [NFKOutputImage],
        "pose-simplebaseline": [NFKOutputPose],
        "retinaface-mobile025": [NFKOutputDetections],
        "rf-detr-nano": [NFKOutputDetections],
        "rf-detr-small": [NFKOutputDetections],
        "rf-detr-seg-small": [NFKOutputDetections, NFKOutputMask],
        "rtdetr-r18vd": [NFKOutputDetections],
        "real-esrgan-general-x4v3": [NFKOutputImage],
        "swinir-lightweight-x2": [NFKOutputImage],
        "swinir-lightweight-x3": [NFKOutputImage],
        "swinir-lightweight-x4": [NFKOutputImage],
    ]
}
