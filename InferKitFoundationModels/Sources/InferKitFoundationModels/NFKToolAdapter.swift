//
//  NFKToolAdapter.swift
//  InferKitFoundationModels
//

import Foundation
import FoundationModels

/// The error a declared tool without a handler throws when the model calls it. The backend reads it
/// as "the caller runs this tool" and ends the turn with the call under `NFKOutputToolCalls`.
struct NFKUnhandledToolCall: Error {
    let name: String
}

/// One tool call the model made during a turn, in the core's `NFKOutputToolCalls` shape.
struct NFKRecordedToolCall: @unchecked Sendable {
    let id: String
    let name: String
    let arguments: [String: Any]
    let argumentsJSON: String

    var dictionary: [String: Any] {
        ["id": id, "name": name, "arguments": arguments, "argumentsJSON": argumentsJSON]
    }
}

/// Collects the calls the model made that no handler ran, across the adapters of one turn.
final class NFKToolCallRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var calls: [NFKRecordedToolCall] = []

    func record(_ call: NFKRecordedToolCall) {
        lock.lock()
        calls.append(call)
        lock.unlock()
    }

    var recorded: [NFKRecordedToolCall] {
        lock.lock()
        defer { lock.unlock() }
        return calls
    }
}

/// Adapts a tool declaration (name, description, JSON Schema parameters) to Apple's `Tool` protocol
/// with a runtime schema, so a consumer registers tools without a compile-time `@Generable` argument
/// type. The model's arguments arrive as `GeneratedContent` and reach the handler as a JSON object.
/// A declaration without a handler is still offered to the model; when called, it records the call
/// and throws, which ends the turn so the caller can run the tool itself.
struct NFKToolAdapter: Tool {

    typealias Arguments = GeneratedContent

    let name: String
    let description: String
    let parameters: GenerationSchema
    private let handler: (@Sendable ([String: Any]) async throws -> String)?
    private let recorder: NFKToolCallRecorder

    init(name: String,
         description: String,
         parameters: [String: Any],
         handler: (@Sendable ([String: Any]) async throws -> String)?,
         recorder: NFKToolCallRecorder) throws {
        self.name = name
        self.description = description
        self.handler = handler
        self.recorder = recorder
        self.parameters = try NFKSchema.generationSchema(name: name + "Arguments", json: parameters)
    }

    init(tool: NFKFoundationTool, recorder: NFKToolCallRecorder) throws {
        try self.init(name: tool.name,
                      description: tool.toolDescription,
                      parameters: tool.parameters,
                      handler: tool.handler,
                      recorder: recorder)
    }

    func call(arguments: GeneratedContent) async throws -> String {
        let values = NFKSchema.dictionary(from: arguments)
        guard let handler else {
            recorder.record(NFKRecordedToolCall(id: UUID().uuidString,
                                                name: name,
                                                arguments: values,
                                                argumentsJSON: arguments.jsonString))
            throw NFKUnhandledToolCall(name: name)
        }
        return try await handler(values)
    }
}
