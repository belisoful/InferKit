//
//  NFKMLXError.swift
//  InferKitMLX
//
//  The package-wide error type. Every file in the package uses these cases, so it has its own.
//

import Foundation

enum NFKMLXError: Error {
    case notReady
    case noOutput
    case unsupportedInput
    /// A checkpoint left some of the model's parameters uncovered; loading it would silently keep them
    /// randomly initialized. See `NFKMLXWeights.apply(_:to:strict:)`.
    case weightsMismatch(String)
    /// A checkpoint could not be written. See `NFKMLXWeights.save(_:to:)`.
    case checkpointNotWritable(String)
    /// A training run's loss stopped being finite, so its parameters are unrecoverable.
    /// See `NFKMLXTrainer.train(_:optimizer:steps:batch:loss:)`.
    case trainingDiverged(String)
    /// Training data does not have the shape the model needs. See `NFKMLXTrainingData`.
    case trainingDataMismatch(String)
    /// A layer selected for low-rank adaptation cannot be replaced. See `NFKMLXLoRA.apply(to:rank:alpha:where:)`.
    case loRANotApplicable(String)
    /// A model was asked for a geometry it has no architecture for, such as a super-resolution scale
    /// the reference upsampler does not build.
    case unsupportedConfiguration(String)
}

extension NFKMLXError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .notReady: return "the MLX backend is not ready"
        case .noOutput: return "the MLX backend produced no output"
        case .unsupportedInput: return "the request does not carry an input this MLX backend supports"
        case .weightsMismatch(let detail): return detail
        case .checkpointNotWritable(let detail): return detail
        case .trainingDiverged(let detail): return detail
        case .trainingDataMismatch(let detail): return detail
        case .loRANotApplicable(let detail): return detail
        case .unsupportedConfiguration(let detail): return detail
        }
    }
}
