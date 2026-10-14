//
//  NFKMLXChatterboxParameterKey.swift
//  InferKitMLX
//
//  The Chatterbox options a request sets, where Objective-C reaches them.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The request-parameter keys for the Chatterbox options that have no core key, so an Objective-C caller
/// sets them per request the way it sets `NFKParameterTemperature`.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXChatterboxParameterKey)
public final class NFKMLXChatterboxParameterKey: NSObject {
    /// How strongly the voice's emotion is exaggerated, as an `NSNumber` float: the reference's
    /// `exaggeration`, whose default is 0.5. It overrides the backend's build-time value for one request.
    /// Introduced in InferKit 0.4.0.
    @objc public static let exaggeration = "NFKMLXParameterChatterboxExaggeration"
}

extension NFKMLXChatterbox {

    /// The request parameters the Chatterbox backend reads beside `NFKParameterSampleRate`.
    static let requestParameterKeys: Set<String> = [
        NFKMLXChatterboxParameterKey.exaggeration, NFKParameterTemperature, NFKParameterGuidanceScale,
        NFKParameterSeed,
    ]

    /// `base` with the request's exaggeration, where the request sets one.
    static func conditionals(_ base: NFKMLXChatterboxConditionals,
                             for request: NFKInferenceRequest) -> NFKMLXChatterboxConditionals {
        guard let value = NFKMLXGenerationBackend.number(request, NFKMLXChatterboxParameterKey.exaggeration) else {
            return base
        }
        var conditionals = base
        conditionals.t3.exaggeration = value.floatValue
        return conditionals
    }

    /// T3's sampling settings: the reference defaults, with the request's temperature, guidance weight,
    /// and seed where it sets them.
    static func samplingOptions(for request: NFKInferenceRequest) -> NFKMLXT3SamplingOptions {
        var options = NFKMLXT3SamplingOptions()
        if let temperature = NFKMLXGenerationBackend.number(request, NFKParameterTemperature) {
            options.temperature = temperature.floatValue
        }
        if let guidance = NFKMLXGenerationBackend.number(request, NFKParameterGuidanceScale) {
            options.guidance = guidance.floatValue
        }
        if let seed = NFKMLXGenerationBackend.number(request, NFKParameterSeed) {
            options.seed = seed.uint64Value
        }
        return options
    }
}
