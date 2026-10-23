//
//  NFKMLXContextRoom.swift
//  InferKitMLX
//
//  The room a model's trained context leaves a reply: a run's token limit bounded by what its prompt left,
//  and a prompt that already fills the context refused.
//

import Foundation

enum NFKMLXContextRoom {

    /// `maxTokens` bounded by the positions `contextLength` leaves after `promptTokens`. A context of nil or 0
    /// bounds nothing.
    ///
    /// - Throws: `NFKMLXError.unsupportedConfiguration` naming both counts when the prompt fills the context.
    static func bounded(_ maxTokens: Int, promptTokens: Int, contextLength: Int?, source: String) throws -> Int {
        guard let contextLength, contextLength > 0 else { return maxTokens }
        guard promptTokens < contextLength else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(source)'s prompt holds \(promptTokens) tokens, which fills its \(contextLength)-token context")
        }
        return min(maxTokens, contextLength - promptTokens)
    }
}
