//
//  NFKMLXCatalogLive.swift
//  InferKitMLX
//
//  The catalog's models that run on a live source, one video frame at a time.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The live profiles of the catalog's models (`NFKModelDescriptor.liveProfile`).
///
/// @discussion A model is declared live when it is built to answer one frame in a bounded single
/// forward: a small segmenter, matter, detector, depth or pose estimator, or upscaler. Each declares the
/// frame's input key and every output key it writes. The models here treat each frame alone.
enum NFKMLXCatalogLive {

    /// The entry with its live profile, where it has one.
    static func live(_ entry: NFKMLXCatalogEntry) -> NFKMLXCatalogEntry {
        guard let outputs = videoFrameOutputs[entry.identifier] else {
            return entry
        }
        var live = entry
        live.liveProfile = NFKModelLiveProfile.videoFrameProfile(withInputKeys: [NFKInputImage], outputKeys: outputs,
                                                                 stateful: false, timings: nil)
        return live
    }

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
