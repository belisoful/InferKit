//
//  NFKVisionSmudgeBackend.swift
//  InferKitAppleSwift
//

import CoreGraphics
import Foundation
import InferKit
import Vision

/// Judges whether a photograph was taken through a smudged lens, with Vision's
/// `DetectLensSmudgeRequest`, which ships only in Vision's Swift module.
///
/// A capture pipeline uses it to tell a person to wipe the lens rather than silently keeping a soft
/// frame, and a batch job uses it to reject frames before spending a model on them.
///
/// - Input: `NFKInputImage`.
/// - Output: one `NFKClassification` under `NFKOutputClassifications`, labeled `smudge`, whose
///   confidence is how sure Vision is that the lens was dirty.
///
/// Needs macOS 26 / iOS 26. Introduced in InferKit 0.4.0.
@objc(NFKVisionSmudgeBackend)
public final class NFKVisionSmudgeBackend: NSObject, NFKInferenceBackend {

    /// How long a request waits for Vision, in seconds. Reaching it is reported as
    /// `kNFKError_InferenceBackendFailure`. Default 60. Introduced in InferKit 0.4.0.
    @objc public var responseTimeout: TimeInterval = 60

    @objc public override init() {
        super.init()
    }

    @objc public var isReady: Bool { true }

    @objc public var backendIdentifier: String { "vision-smudge" }

    @objc public var supportedInputKeys: Set<String> { [NFKInputImage] }

    @objc public var supportedOutputKeys: Set<String> { [NFKOutputClassifications] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let image = NFKAppleImage.cgImage(in: request, key: NFKInputImage) else {
            throw NFKAppleImage.error(.error_InferenceMissingInput,
                                      "the request carries no image under NFKInputImage")
        }

        let outcome = NFKAppleOutcome<NFKInferenceResult>()
        let semaphore = DispatchSemaphore(value: 0)
        let task = Task.detached(priority: .userInitiated) {
            do {
                let observation = try await DetectLensSmudgeRequest().perform(on: image)
                let smudge = NFKClassification(label: "smudge",
                                               classIndex: 0,
                                               confidence: Double(observation.confidence))
                outcome.succeed(NFKInferenceResult(outputs: [NFKOutputClassifications: [smudge]]))
            } catch {
                outcome.fail(error)
            }
            semaphore.signal()
        }
        try NFKAppleWait.wait(semaphore, seconds: responseTimeout, cancelling: task,
                              code: .error_InferenceBackendFailure, reason: "Vision did not finish judging the lens")
        return try outcome.value()
    }
}
