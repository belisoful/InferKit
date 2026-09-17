//
//  NFKFoundationModel.swift
//  InferKitFoundationModels
//

import Foundation
import FoundationModels
import InferKit

/// The Apple model a `NFKFoundationModelsBackend` runs. Introduced in InferKit 0.4.0.
@objc public enum NFKFoundationModel: Int, Sendable {
    /// The on-device system language model (`SystemLanguageModel`). Needs Apple Intelligence.
    case onDevice = 0
    /// Apple's larger model on Private Cloud Compute (`PrivateCloudComputeLanguageModel`). Needs
    /// macOS 27 / iOS 27; the request leaves the device, and usage counts against a quota.
    case privateCloudCompute = 1
}

/// The specialization of the on-device model. Introduced in InferKit 0.4.0.
@objc public enum NFKFoundationModelUseCase: Int, Sendable {
    /// General text generation (`SystemLanguageModel.UseCase.general`).
    case general = 0
    /// Tagging topics, entities, emotions, and actions (`SystemLanguageModel.UseCase.contentTagging`).
    case contentTagging = 1
}

/// The content guardrails the on-device model applies. Introduced in InferKit 0.4.0.
@objc public enum NFKFoundationModelGuardrails: Int, Sendable {
    /// Apple's default guardrails (`SystemLanguageModel.Guardrails.default`).
    case `default` = 0
    /// Guardrails that allow transforming content the caller supplies, such as summarizing or
    /// rewriting it (`SystemLanguageModel.Guardrails.permissiveContentTransformations`).
    case permissiveContentTransformations = 1
}

/// A reading of the Private Cloud Compute quota. Introduced in InferKit 0.4.0.
///
/// The values are a snapshot; read ``NFKFoundationModelsBackend/privateCloudComputeQuota`` again for
/// a current one. An app that sees `isLimitReached` or `isApproachingLimit` switches the backend's
/// `model` to `.onDevice`, or hands the request to an `NFKRemoteBackend`, before the quota error.
@available(macOS 27, iOS 27, *)
@objc(NFKFoundationModelQuota)
public final class NFKFoundationModelQuota: NSObject {

    /// The quota is used up; a request fails until `resetDate`.
    @objc public let isLimitReached: Bool

    /// The quota is close to its limit.
    @objc public let isApproachingLimit: Bool

    /// When the quota resets, where the service reports it.
    @objc public let resetDate: Date?

    /// The system offers a way to raise the limit; `showLimitIncreaseSuggestion()` presents it.
    @objc public var canShowLimitIncreaseSuggestion: Bool {
        limitIncreaseSuggestion != nil
    }

    // Boxed: an `@objc` class is realized when the binary loads, which lays out its stored
    // properties and needs their types' metadata, and the suggestion's type does not exist below
    // macOS 27.
    private let limitIncreaseSuggestion: Any?

    init(isLimitReached: Bool, isApproachingLimit: Bool, resetDate: Date?, limitIncreaseSuggestion: Any?) {
        self.isLimitReached = isLimitReached
        self.isApproachingLimit = isApproachingLimit
        self.resetDate = resetDate
        self.limitIncreaseSuggestion = limitIncreaseSuggestion
        super.init()
    }

    #if compiler(>=6.4)
    convenience init(usage: PrivateCloudComputeLanguageModel.QuotaUsage) {
        let isLimitReached: Bool
        let isApproachingLimit: Bool
        switch usage.status {
        case .belowLimit(let status):
            isLimitReached = false
            isApproachingLimit = status.isApproachingLimit
        case .limitReached:
            isLimitReached = true
            isApproachingLimit = true
        @unknown default:
            isLimitReached = false
            isApproachingLimit = false
        }
        self.init(isLimitReached: isLimitReached,
                  isApproachingLimit: isApproachingLimit,
                  resetDate: usage.resetDate,
                  limitIncreaseSuggestion: usage.limitIncreaseSuggestion)
    }
    #endif

    /// Presents the system's limit-increase suggestion. Returns `false` when there is none.
    @objc @discardableResult
    public func showLimitIncreaseSuggestion() -> Bool {
        #if compiler(>=6.4)
        guard let suggestion = limitIncreaseSuggestion
                as? PrivateCloudComputeLanguageModel.QuotaUsage.LimitIncreaseSuggestion else { return false }
        suggestion.show()
        return true
        #else
        return false
        #endif
    }
}

extension NFKFoundationModelUseCase {
    var systemUseCase: SystemLanguageModel.UseCase {
        switch self {
        case .general: return .general
        case .contentTagging: return .contentTagging
        }
    }
}

extension NFKFoundationModelGuardrails {
    var systemGuardrails: SystemLanguageModel.Guardrails {
        switch self {
        case .default: return .default
        case .permissiveContentTransformations: return .permissiveContentTransformations
        }
    }
}

/// The model a backend is set to, captured when a request is submitted so a later change to the
/// backend does not move a running request.
struct NFKFoundationModelConfiguration: Equatable, Sendable {
    var model: NFKFoundationModel = .onDevice
    var useCase: NFKFoundationModelUseCase = .general
    var guardrails: NFKFoundationModelGuardrails = .default

    /// The on-device model with the configured specialization and guardrails.
    var systemModel: SystemLanguageModel {
        if useCase == .general, guardrails == .default {
            return .default
        }
        return SystemLanguageModel(useCase: useCase.systemUseCase, guardrails: guardrails.systemGuardrails)
    }

    var isReady: Bool {
        (try? checkAvailability()) != nil
    }

    /// Throws the core's error when the configured model cannot take a request: the on-device model
    /// is unavailable, Private Cloud Compute is unavailable or over quota, or the OS predates it.
    func checkAvailability() throws {
        switch model {
        case .onDevice:
            try checkSystemAvailability()
        case .privateCloudCompute:
            #if compiler(>=6.4)
            guard #available(macOS 27, iOS 27, *) else {
                throw Self.privateCloudComputeUnsupported()
            }
            try checkPrivateCloudComputeAvailability()
            #else
            throw Self.privateCloudComputeUnsupported()
            #endif
        }
    }

    private func checkSystemAvailability() throws {
        switch systemModel.availability {
        case .available:
            return
        case .unavailable(let reason):
            throw Self.notReady("the system language model is unavailable: \(reason)")
        @unknown default:
            throw Self.notReady("the system language model is unavailable")
        }
    }

    #if compiler(>=6.4)
    @available(macOS 27, iOS 27, *)
    private func checkPrivateCloudComputeAvailability() throws {
        let cloud = PrivateCloudComputeLanguageModel()
        switch cloud.availability {
        case .available:
            break
        case .unavailable(let reason):
            throw Self.notReady("Private Cloud Compute is unavailable: \(reason)")
        @unknown default:
            throw Self.notReady("Private Cloud Compute is unavailable")
        }
        let quota = cloud.quotaUsage
        if case .limitReached = quota.status {
            var userInfo: [String: Any] = [NSLocalizedDescriptionKey: "the Private Cloud Compute quota is reached"]
            if let resetDate = quota.resetDate {
                userInfo[NFKFoundationModelsErrorKey.resetDate] = resetDate
            }
            throw NSError(domain: NFKInferenceErrorDomain,
                          code: NFKInferenceError.error_InferenceNotReady.rawValue,
                          userInfo: userInfo)
        }
    }
    #endif

    /// A session over the configured model, seeded with the transcript entries and offering the tools.
    func makeSession(tools: [any Tool], entries: [Transcript.Entry]) throws -> LanguageModelSession {
        let transcript = Transcript(entries: entries)
        switch model {
        case .onDevice:
            return LanguageModelSession(model: systemModel, tools: tools, transcript: transcript)
        case .privateCloudCompute:
            #if compiler(>=6.4)
            guard #available(macOS 27, iOS 27, *) else {
                throw Self.privateCloudComputeUnsupported()
            }
            return LanguageModelSession(model: PrivateCloudComputeLanguageModel(), tools: tools, transcript: transcript)
            #else
            throw Self.privateCloudComputeUnsupported()
            #endif
        }
    }

    private static func notReady(_ description: String) -> NSError {
        NSError(domain: NFKInferenceErrorDomain,
                code: NFKInferenceError.error_InferenceNotReady.rawValue,
                userInfo: [NSLocalizedDescriptionKey: description])
    }

    private static func privateCloudComputeUnsupported() -> NSError {
        NSError(domain: NFKInferenceErrorDomain,
                code: NFKInferenceError.error_InferenceUnsupported.rawValue,
                userInfo: [NSLocalizedDescriptionKey: "Private Cloud Compute needs macOS 27 / iOS 27 and a build with the macOS 27 SDK"])
    }
}
