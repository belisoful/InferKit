//
//  NFKMLXStepSampling.swift
//  InferKitMLX
//
//  How a decoder that runs its own token loop picks each token: greedy at temperature 0, else drawn at the
//  temperature from the logits the top-k and nucleus filters leave, as the shared sampler draws.
//

import Foundation
import InferKit
import MLX

struct NFKMLXStepSampling {
    var temperature: Float = 0
    var topP: Float = 1
    var topK = 0

    /// The sampling a release's generation config sets, or greedy decoding without one: steps 3 and 4 of the
    /// sampling order ``NFKMLXGenerationOptions`` states, which ``read(_:)`` completes with step 1.
    init(release: NFKMLXReleaseSampling?) {
        if let release {
            temperature = release.temperature
            topP = release.topP
            topK = release.topK
        }
    }

    /// Takes the request's NFKParameterTemperature, NFKParameterTopP, and NFKParameterTopK over these values.
    mutating func read(_ request: NFKInferenceRequest) {
        if let value = request.parameter(forKey: NFKParameterTemperature) as? NSNumber {
            temperature = value.floatValue
        }
        if let value = request.parameter(forKey: NFKParameterTopP) as? NSNumber {
            topP = value.floatValue
        }
        if let value = request.parameter(forKey: NFKParameterTopK) as? NSNumber {
            topK = value.intValue
        }
    }

    /// The next token from one row of logits, the random draw seeded with `seed` plus the step where a seed
    /// is set. Without a filter the draw is the one these decoders always made, so a seeded run repeats.
    func next(_ logits: MLXArray, seed: UInt64?, step: Int) -> Int {
        guard temperature > 0 else {
            return logits.argMax(axis: -1).item(Int.self)
        }
        if let seed {
            MLXRandom.seed(seed &+ UInt64(step))
        }
        guard topP < 1 || topK > 0 else {
            return MLXRandom.categorical(logits * (1 / temperature)).item(Int.self)
        }
        var options = NFKMLXGenerationOptions()
        options.temperature = temperature
        options.topP = topP
        options.topK = topK
        return MLXRandom.categorical(log(NFKMLXLanguageNet.probabilities(of: logits, options: options))).item(Int.self)
    }
}
