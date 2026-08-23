//
//  NFKToolAdapter.swift
//  InferKitFoundationModels
//

import Foundation
import FoundationModels

/// Adapts an `NFKFoundationTool` to Apple's `Tool` protocol with a runtime-defined schema, so a
/// consumer registers tools without a compile-time `@Generable` argument type. The model's arguments
/// arrive as `GeneratedContent`; this reads the declared parameters into a dictionary and hands them
/// to the tool's handler.
struct NFKToolAdapter: Tool {

    typealias Arguments = GeneratedContent

    let name: String
    let description: String
    let parameters: GenerationSchema
    private let specifications: [NFKFoundationToolParameter]
    private let handler: @Sendable ([String: Any]) async throws -> String

    init(tool: NFKFoundationTool) throws {
        self.name = tool.name
        self.description = tool.toolDescription
        self.specifications = tool.parameters
        self.handler = tool.handler
        self.parameters = try NFKSchema.generationSchema(name: tool.name + "Arguments",
                                                         description: tool.toolDescription,
                                                         properties: tool.parameters)
    }

    func call(arguments: GeneratedContent) async throws -> String {
        var values: [String: Any] = [:]
        for specification in specifications {
            do {
                values[specification.name] = try NFKSchema.value(of: specification.type,
                                                                 from: arguments,
                                                                 property: specification.name)
            } catch {
                if specification.isRequired {
                    throw error
                }
            }
        }
        return try await handler(values)
    }
}
