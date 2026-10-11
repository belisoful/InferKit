//
//  NFKFoundationTool.swift
//  InferKitFoundationModels
//

import Foundation
import FoundationModels
import InferKit

/// A tool the on-device model can call during generation, with the handler that runs it.
///
/// Register tools on `NFKFoundationModelsBackend.tools`. A tool declares itself the way the core's
/// `NFKParameterTools` entries do: a name, a description the model reads to decide relevance, and a
/// JSON Schema object for its arguments. A request that carries `NFKParameterTools` declares its own
/// tool set and reaches these by name for their handlers; a request without it offers every
/// registered tool. The handler receives the model's arguments as the parsed JSON object (`String`,
/// `NSNumber`, `Bool`, nested `[String: Any]` and `[Any]`) and returns the tool's result text, which
/// the model reads before continuing its reply. Introduced in InferKit 0.4.0.
///
/// The tool crosses between Apple's tool subsystem and the core's in both directions:
/// - ``init(tool:)`` puts any core `NFKInferenceTool` on Apple's model, an `NFKBackendTool` among
///   them, so the system model calls a local language, speech, or image model.
/// - ``init(appleTool:)`` takes an app's own Foundation Models `Tool`, and ``inferenceTool`` presents
///   any of these tools as an `NFKInferenceTool`, so `NFKToolLoop` offers it to a remote or local
///   model and `NFKInferenceServer` serves it over MCP.
@objc(NFKFoundationTool)
public final class NFKFoundationTool: NSObject, @unchecked Sendable {

    @objc public let name: String
    @objc public let toolDescription: String
    /// The JSON Schema object for the arguments, the same shape as an `NFKParameterTools` entry's
    /// `parameters`: `{"type": "object", "properties": {…}, "required": […]}`.
    @objc public let parameters: [String: Any]
    /// Runs the tool to the result the backend records; the model reads its text.
    let resultHandler: @Sendable ([String: Any]) async throws -> NFKToolResult

    /// The result's text alone.
    var handler: @Sendable ([String: Any]) async throws -> String {
        let resultHandler = self.resultHandler
        return { try await resultHandler($0).text }
    }

    /// How long ``inferenceTool`` waits for the handler, in seconds. Defaults to 600. A handler that
    /// has not answered by then is reported as failed, so a call never hangs its caller.
    @objc public var timeout: TimeInterval = 600

    /// Swift: an asynchronous handler (for tools that do I/O).
    public init(name: String,
                description: String,
                parameters: [String: Any],
                handler: @escaping @Sendable ([String: Any]) async throws -> String) {
        self.name = name
        self.toolDescription = description
        self.parameters = parameters
        self.resultHandler = { arguments in NFKToolResult(text: try await handler(arguments)) }
        super.init()
    }

    /// Objective-C: a synchronous handler.
    @objc public init(name: String,
                      description: String,
                      parameters: [String: Any],
                      syncHandler: @escaping @Sendable ([String: Any]) -> String) {
        self.name = name
        self.toolDescription = description
        self.parameters = parameters
        self.resultHandler = { arguments in NFKToolResult(text: syncHandler(arguments)) }
        super.init()
    }

    /// A core tool on Apple's model: its name, description, and schema, run through its own
    /// `invokeWithArguments:`, which checks the arguments first. The tool runs off the Swift
    /// concurrency pool, since a model behind it blocks for seconds. Its outputs, such as the audio a
    /// speech tool generated, come back under `NFKOutputToolInvocations`. Introduced in InferKit 0.4.0.
    @objc(initWithInferenceTool:)
    public init(tool: NFKInferenceTool) {
        self.name = tool.name
        self.toolDescription = tool.toolDescription
        self.parameters = tool.parameters
        let box = NFKToolBox(tool)
        self.resultHandler = { arguments in
            await withCheckedContinuation { continuation in
                DispatchQueue.global(qos: .userInitiated).async {
                    continuation.resume(returning: box.tool.invoke(withArguments: arguments))
                }
            }
        }
        super.init()
    }

    /// An app's own Foundation Models tool, so an `@Generable` tool written for Apple's model reaches
    /// every other model through ``inferenceTool``. The schema is its `GenerationSchema` encoded as a
    /// JSON Schema; the model's JSON arguments become its `Arguments` through `GeneratedContent`; a
    /// `String` output is the result text, and generated content its JSON. Swift only: `Tool` is a
    /// Swift protocol with associated types. Introduced in InferKit 0.4.0.
    public convenience init<AppleTool: Tool>(appleTool: AppleTool) throws {
        let schema = try JSONSerialization.jsonObject(with: try JSONEncoder().encode(appleTool.parameters)) as? [String: Any]
        self.init(name: appleTool.name, description: appleTool.description, parameters: schema ?? ["type": "object"],
                  resultHandler: { arguments in
            let json = try JSONSerialization.data(withJSONObject: arguments)
            let content = try GeneratedContent(json: String(decoding: json, as: UTF8.self))
            let output = try await appleTool.call(arguments: try AppleTool.Arguments(content))
            return NFKToolResult(text: Self.text(of: output))
        })
    }

    private init(name: String, description: String, parameters: [String: Any],
                 resultHandler: @escaping @Sendable ([String: Any]) async throws -> NFKToolResult) {
        self.name = name
        self.toolDescription = description
        self.parameters = parameters
        self.resultHandler = resultHandler
        super.init()
    }

    static func text(of output: some PromptRepresentable) -> String {
        if let text = output as? String { return text }
        if let content = output as? any ConvertibleToGeneratedContent { return content.generatedContent.jsonString }
        return String(describing: output)
    }

    /// The tool as an `NFKParameterTools` entry, so a request built for a remote backend and one
    /// built from registered tools carry the same dictionary.
    @objc public var declaration: [String: Any] {
        ["name": name, "description": toolDescription, "parameters": parameters]
    }

    /// The tool as a core `NFKInferenceTool`, for `NFKToolLoop`, `NFKInferenceServer.addTool:`, and
    /// any model a loop drives. A call waits on the handler at most ``timeout`` seconds, and a handler
    /// that throws is an error result the calling model reads. Introduced in InferKit 0.4.0.
    @objc public var inferenceTool: NFKInferenceTool {
        let handler = resultHandler
        let timeout = self.timeout
        let name = self.name
        return NFKInferenceTool(name: name, description: toolDescription, parameters: parameters) { arguments, error in
            let outcome = NFKToolOutcome()
            let finished = DispatchSemaphore(value: 0)
            Task.detached(priority: .userInitiated) {
                do {
                    outcome.result = try await handler(arguments)
                } catch {
                    outcome.result = NFKToolResult.errorResult(withText: "\(name) failed: \(error.localizedDescription)")
                }
                finished.signal()
            }
            guard finished.wait(timeout: .now() + timeout) == .success, let result = outcome.result else {
                return NFKToolResult.errorResult(withText: "\(name) did not answer within \(Int(timeout)) seconds")
            }
            return result
        }
    }
}

/// A core tool carried into a `@Sendable` handler. `invokeWithArguments:` may be called from several
/// threads at once, which the core's tool contract states.
private final class NFKToolBox: @unchecked Sendable {
    let tool: NFKInferenceTool
    init(_ tool: NFKInferenceTool) { self.tool = tool }
}

/// What a detached tool run hands back across the semaphore.
private final class NFKToolOutcome: @unchecked Sendable {
    var result: NFKToolResult?
}
