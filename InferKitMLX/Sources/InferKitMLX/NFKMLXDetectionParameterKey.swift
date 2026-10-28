//
//  NFKMLXDetectionParameterKey.swift
//  InferKitMLX
//
//  The request parameters the detectors and the voice activity detectors read, so a request, and through it
//  a catalog option, sets the thresholds a backend applies.
//

import Foundation
import InferKit

/// The request parameters the object and face detectors read: YOLO, RT-DETR, RF-DETR, the Table
/// Transformer, and RetinaFace. Each overrides the backend's own value for one request.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXDetectionParameterKey)
public final class NFKMLXDetectionParameterKey: NSObject {
    /// The score a detection must reach to be returned, as an `NSNumber` in `0...1`. The default is the
    /// model's own: 0.25 for YOLO, 0.3 for RT-DETR and RF-DETR, 0.6 for the Table Transformer, and 0.8 for
    /// RetinaFace. Introduced in InferKit 0.4.0.
    @objc public static let confidenceThreshold = "NFKMLXParameterDetectionConfidenceThreshold"
    /// The overlap, as intersection over union in `0...1`, above which non-max suppression drops the
    /// less confident of two boxes of the same class. Only the detectors that suppress read it: YOLOv8,
    /// YOLOv9, YOLO11, and YOLO12 (0.7), and RetinaFace (0.4). YOLOv10 and YOLO26, which are end to end,
    /// and the DETR family emit one box per object and do not. Introduced in InferKit 0.4.0.
    @objc public static let overlapThreshold = "NFKMLXParameterDetectionOverlapThreshold"

    /// The request's value under `key` as a `Float`, else `fallback`.
    static func threshold(_ request: NFKInferenceRequest, _ key: String, default fallback: Float) -> Float {
        NFKMLXGenerationBackend.number(request, key)?.floatValue ?? fallback
    }
}

/// The request parameters the voice activity detectors read: Silero VAD and MarbleNet.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXVoiceActivityParameterKey)
public final class NFKMLXVoiceActivityParameterKey: NSObject {
    /// The speech probability a frame must reach to count as speech, as an `NSNumber` in `0...1` (default
    /// 0.5). Consecutive frames at or above it form one span. It overrides the configuration's
    /// `threshold` for one request. Introduced in InferKit 0.4.0.
    @objc public static let speechThreshold = "NFKMLXParameterVoiceActivitySpeechThreshold"
}
