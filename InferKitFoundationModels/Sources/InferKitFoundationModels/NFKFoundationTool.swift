//
//  NFKFoundationTool.swift
//  InferKitFoundationModels
//

import Foundation

/// The type of a tool argument the model fills in.
@objc(NFKToolParameterType)
public enum NFKToolParameterType: Int, Sendable {
    case string
    case integer
    case number
    case boolean
}

/// One argument a tool accepts: its name, a description that guides the model, its type, and whether
/// the model must supply it.
@objc(NFKFoundationToolParameter)
public final class NFKFoundationToolParameter: NSObject, Sendable {

    @objc public let name: String
    @objc public let parameterDescription: String
    @objc public let type: NFKToolParameterType
    @objc public let isRequired: Bool

    @objc public init(name: String, description: String, type: NFKToolParameterType, required: Bool) {
        self.name = name
        self.parameterDescription = description
        self.type = type
        self.isRequired = required
        super.init()
    }
}

/// A tool the on-device model can call during generation. Register tools on
/// `NFKFoundationModelsBackend.tools`; the model decides when to call one based on its name,
/// description, and parameters. The handler receives the model's arguments as a dictionary keyed by
/// parameter name (values are `String`, `Int`, `Double`, or `Bool`) and returns the tool's result
/// text, which the model reads before continuing its reply.
@objc(NFKFoundationTool)
public final class NFKFoundationTool: NSObject, @unchecked Sendable {

    @objc public let name: String
    @objc public let toolDescription: String
    @objc public let parameters: [NFKFoundationToolParameter]
    let handler: @Sendable ([String: Any]) async throws -> String

    /// Swift: an asynchronous handler (for tools that do I/O).
    public init(name: String,
                description: String,
                parameters: [NFKFoundationToolParameter],
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
                      parameters: [NFKFoundationToolParameter],
                      syncHandler: @escaping @Sendable ([String: Any]) -> String) {
        self.name = name
        self.toolDescription = description
        self.parameters = parameters
        self.handler = { arguments in syncHandler(arguments) }
        super.init()
    }
}
