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

    /// The entry with its live profile, where it has one.
    static func live(_ entry: NFKMLXCatalogEntry) -> NFKMLXCatalogEntry {
        if let audio = audioChunks[entry.identifier] {
            var live = entry
            live.liveProfile = audio
            return live
        }
        let stateful = statefulVideoFrameOutputs[entry.identifier] != nil
        guard let outputs = videoFrameOutputs[entry.identifier] ?? statefulVideoFrameOutputs[entry.identifier] else {
            return entry
        }
        var live = entry
        live.liveProfile = NFKModelLiveProfile.videoFrameProfile(withInputKeys: [NFKInputImage], outputKeys: outputs,
                                                                 stateful: stateful, timings: nil)
        return live
    }

    /// The audio-chunk profile of each audio model whose backend opens a live stream.
    static let audioChunks: [String: NFKModelLiveProfile] = [
        "silero-vad": .audioChunkProfile(withInputKeys: [NFKInputAudio], outputKeys: [NFKOutputSegments], stateful: true,
                                         chunkDuration: 0.032, sampleRate: 16000, channelCount: 1,
                                         lookaheadMilliseconds: 0, timings: nil),
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
