//
//  NFKFoundationModelsErrors.swift
//  InferKitFoundationModels
//

import Foundation
import FoundationModels
import InferKit

/// Turns a Foundation Models failure into the core's error, so a consumer reads a failure from
/// Apple's model the way it reads one from a remote endpoint or an MLX model.
///
/// The framework's own error is kept under `NSUnderlyingErrorKey` for a caller that wants the
/// original. What the core code carries is the decision an app has to make: retry later, change the
/// request, or tell the user.
enum NFKFoundationModelsFailure {

    /// The core error a failure becomes. An error the backend itself raised passes through, since it
    /// is already in the core's domain.
    static func coreError(for error: Error) -> NSError {
        let raised = error as NSError
        if raised.domain == NFKInferenceErrorDomain {
            return raised
        }

        #if compiler(>=6.4)
        if #available(macOS 27, iOS 27, *), let mapped = modernError(for: error) {
            return mapped
        }
        #endif
        if let generation = error as? LanguageModelSession.GenerationError {
            return coreError(for: generation)
        }
        return coreError(code: NFKInferenceError.error_InferenceBackendFailure.rawValue, error: error)
    }

    // MARK: macOS 26

    /// `GenerationError` is what the framework raises below macOS 27, and what it still raises on 27
    /// for a session built the old way. Its cases carry a description and nothing else, so the
    /// counts a 27 error would give are absent here.
    private static func coreError(for error: LanguageModelSession.GenerationError) -> NSError {
        switch error {
        case .exceededContextWindowSize:
            return coreError(code: NFKInferenceError.error_InferenceUnsupported.rawValue, error: error)
        case .assetsUnavailable:
            return coreError(code: NFKInferenceError.error_InferenceNotReady.rawValue, error: error)
        case .guardrailViolation, .refusal:
            return coreError(code: NFKInferenceError.error_InferenceRefused.rawValue, error: error)
        case .unsupportedGuide, .unsupportedLanguageOrLocale:
            return coreError(code: NFKInferenceError.error_InferenceUnsupported.rawValue, error: error)
        case .rateLimited:
            return coreError(code: NFKInferenceError.error_InferenceRateLimited.rawValue, error: error)
        case .decodingFailure, .concurrentRequests:
            return coreError(code: NFKInferenceError.error_InferenceBackendFailure.rawValue, error: error)
        @unknown default:
            return coreError(code: NFKInferenceError.error_InferenceBackendFailure.rawValue, error: error)
        }
    }

    // MARK: macOS 27

    #if compiler(>=6.4)

    /// The errors the 27 SDK introduced, which carry the counts and the reset date the core keys
    /// were made for. Returns nil for an error from none of these types.
    @available(macOS 27, iOS 27, *)
    private static func modernError(for error: Error) -> NSError? {
        if let model = error as? LanguageModelError {
            return coreError(for: model)
        }
        if let cloud = error as? PrivateCloudComputeLanguageModel.Error {
            return coreError(for: cloud)
        }
        if let system = error as? SystemLanguageModel.Error {
            return coreError(code: NFKInferenceError.error_InferenceNotReady.rawValue, error: system)
        }
        if let session = error as? LanguageModelSession.Error {
            return coreError(code: NFKInferenceError.error_InferenceBackendFailure.rawValue, error: session)
        }
        return nil
    }

    @available(macOS 27, iOS 27, *)
    private static func coreError(for error: LanguageModelError) -> NSError {
        switch error {
        case .contextSizeExceeded(let exceeded):
            return coreError(code: NFKInferenceError.error_InferenceUnsupported.rawValue,
                             error: error,
                             userInfo: [NFKFoundationModelsErrorKey.tokenCount: exceeded.tokenCount,
                                        NFKFoundationModelsErrorKey.contextSize: exceeded.contextSize])
        case .rateLimited(let limited):
            var userInfo: [String: Any] = [:]
            if let resetDate = limited.resetDate {
                userInfo[NFKFoundationModelsErrorKey.resetDate] = resetDate
            }
            return coreError(code: NFKInferenceError.error_InferenceRateLimited.rawValue,
                             error: error,
                             userInfo: userInfo)
        case .guardrailViolation, .refusal:
            return coreError(code: NFKInferenceError.error_InferenceRefused.rawValue, error: error)
        case .unsupportedCapability, .unsupportedTranscriptContent,
             .unsupportedGenerationGuide, .unsupportedLanguageOrLocale:
            return coreError(code: NFKInferenceError.error_InferenceUnsupported.rawValue, error: error)
        case .timeout:
            return coreError(code: NFKInferenceError.error_InferenceBackendFailure.rawValue, error: error)
        @unknown default:
            return coreError(code: NFKInferenceError.error_InferenceBackendFailure.rawValue, error: error)
        }
    }

    @available(macOS 27, iOS 27, *)
    private static func coreError(for error: PrivateCloudComputeLanguageModel.Error) -> NSError {
        switch error {
        case .networkFailure, .serviceUnavailable:
            // The service answered with nothing, which is what the core's unreachable code is for.
            return coreError(code: NFKInferenceError.error_RemoteUnreachable.rawValue, error: error)
        case .quotaLimitReached(let reached):
            var userInfo: [String: Any] = [:]
            if let resetDate = reached.resetDate {
                userInfo[NFKFoundationModelsErrorKey.resetDate] = resetDate
            }
            return coreError(code: NFKInferenceError.error_InferenceRateLimited.rawValue,
                             error: error,
                             userInfo: userInfo)
        @unknown default:
            return coreError(code: NFKInferenceError.error_InferenceBackendFailure.rawValue, error: error)
        }
    }

    #endif

    // MARK: Building

    private static func coreError(code: Int,
                                  error: Error,
                                  userInfo: [String: Any] = [:]) -> NSError {
        var info = userInfo
        info[NSLocalizedDescriptionKey] = description(of: error)
        info[NSUnderlyingErrorKey] = error as NSError
        return NSError(domain: NFKInferenceErrorDomain, code: code, userInfo: info)
    }

    /// The framework writes the reason a person can read into `errorDescription`, and the detail a
    /// maintainer needs into the description.
    private static func description(of error: Error) -> String {
        if let localized = (error as? LocalizedError)?.errorDescription, !localized.isEmpty {
            return localized
        }
        return String(describing: error)
    }
}
