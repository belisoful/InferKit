//
//  NFKFoundationTool.swift
//  InferKitFoundationModels
//

import Foundation

/// A tool the on-device model can call during generation, with the handler that runs it.
///
/// Register tools on `NFKFoundationModelsBackend.tools`. A tool declares itself the way the core's
/// `NFKParameterTools` entries do: a name, a description the model reads to decide relevance, and a
/// JSON Schema object for its arguments. A request that carries `NFKParameterTools` declares its own
/// tool set and reaches these by name for their handlers; a request without it offers every
/// registered tool. The handler receives the model's arguments as the parsed JSON object (`String`,
/// `NSNumber`, `Bool`, nested `[String: Any]` and `[Any]`) and returns the tool's result text, which
/// the model reads before continuing its reply. Introduced in InferKit 0.4.0.
@objc(NFKFoundationTool)
public final class NFKFoundationTool: NSObject, @unchecked Sendable {

    @objc public let name: String
    @objc public let toolDescription: String
    /// The JSON Schema object for the arguments, the same shape as an `NFKParameterTools` entry's
    /// `parameters`: `{"type": "object", "properties": {…}, "required": […]}`.
    @objc public let parameters: [String: Any]
    let handler: @Sendable ([String: Any]) async throws -> String

    /// Swift: an asynchronous handler (for tools that do I/O).
    public init(name: String,
                description: String,
                parameters: [String: Any],
                handler: @escaping @Sendable ([String: Any]) async throws -> String) {
        self.name = name
        self.toolDescription = description
        self.parameters = parameters
        self.handler = handler
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
        self.handler = { arguments in syncHandler(arguments) }
        super.init()
    }

    /// The tool as an `NFKParameterTools` entry, so a request built for a remote backend and one
    /// built from registered tools carry the same dictionary.
    @objc public var declaration: [String: Any] {
        ["name": name, "description": toolDescription, "parameters": parameters]
    }
}
