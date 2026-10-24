//
//  NFKMLXContextRoom.swift
//  InferKitMLX
//
//  The room a model's trained context leaves a reply: a run's token limit bounded by what its prompt left, an
//  unset limit run to that room, and a prompt that already fills the context refused.
//

import Foundation
import InferKit

enum NFKMLXContextRoom {

    /// The most tokens a release that states no context runs a reply to when the request sets no limit.
    static let unstatedContextLimit = 256

    /// A run's token limit:
    /// - `maxTokens` set → it, bounded by the positions `contextLength` leaves after `promptTokens`.
    /// - `maxTokens` nil → the positions `window` leaves after the prompt where the cache keeps a window the
    ///   prompt does not fill, and otherwise the positions `contextLength` leaves.
    /// - `contextLength` nil or 0 → `maxTokens`, or ``unstatedContextLimit``.
    ///
    /// - Throws: `NFKMLXError.unsupportedConfiguration` naming both counts when the prompt fills the context.
    static func bounded(_ maxTokens: Int?, promptTokens: Int, contextLength: Int?, window: Int? = nil,
                        source: String) throws -> Int {
        guard let contextLength, contextLength > 0 else { return maxTokens ?? unstatedContextLimit }
        guard promptTokens < contextLength else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(source)'s prompt holds \(promptTokens) tokens, which fills its \(contextLength)-token context")
        }
        let room = contextLength - promptTokens
        if let maxTokens {
            return min(maxTokens, room)
        }
        if let window, window > promptTokens {
            return min(window - promptTokens, room)
        }
        return room
    }

    /// Why a run stopped, as `NFKOutputFinishReason` states it: tool calls where `outputs` call tools,
    /// length where the run produced its `limit`, and stop otherwise.
    static func finishReason(produced: Int, limit: Int, outputs: [String: Any] = [:]) -> String {
        if (outputs[NFKOutputToolCalls] as? [Any])?.isEmpty == false {
            return NFKFinishReasonToolCalls
        }
        return produced >= limit ? NFKFinishReasonLength : NFKFinishReasonStop
    }
}

extension NFKMLXLanguageConfiguration {

    /// The positions a prompt and its reply may fill: the trained window, or the window `ropeScaling`
    /// stretches it to (`original_max_position_embeddings` times `factor`) where that is longer. Nil where
    /// the release states no window.
    var contextLength: Int? {
        guard let maximumPositions else { return nil }
        guard let scaling = ropeScaling else { return maximumPositions }
        return max(maximumPositions, Int(Float(scaling.originalMaxPositionEmbeddings) * scaling.factor))
    }
}
