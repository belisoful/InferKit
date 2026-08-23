//
//  NFKSchema.swift
//  InferKitFoundationModels
//

import Foundation
import FoundationModels

/// Builds Foundation Models schemas from InferKit's runtime field descriptions and reads results
/// back out. Shared by tool arguments (`NFKToolAdapter`) and structured output.
enum NFKSchema {

    /// A `GenerationSchema` for a named object with the given typed properties.
    static func generationSchema(name: String,
                                 description: String,
                                 properties: [NFKFoundationToolParameter]) throws -> GenerationSchema {
        let dynamicProperties = properties.map { property in
            DynamicGenerationSchema.Property(name: property.name,
                                             description: property.parameterDescription,
                                             schema: leafSchema(for: property.type),
                                             isOptional: !property.isRequired)
        }
        let root = DynamicGenerationSchema(name: name, description: description, properties: dynamicProperties)
        return try GenerationSchema(root: root, dependencies: [])
    }

    /// Reads the declared properties out of a generated object into a dictionary keyed by name.
    /// A missing optional property is left out; a missing required property is left out too, since
    /// the model was asked to supply it.
    static func dictionary(from content: GeneratedContent,
                           properties: [NFKFoundationToolParameter]) -> [String: Any] {
        var values: [String: Any] = [:]
        for property in properties {
            if let value = try? value(of: property.type, from: content, property: property.name) {
                values[property.name] = value
            }
        }
        return values
    }

    static func leafSchema(for type: NFKToolParameterType) -> DynamicGenerationSchema {
        switch type {
        case .string:  return DynamicGenerationSchema(type: String.self)
        case .integer: return DynamicGenerationSchema(type: Int.self)
        case .number:  return DynamicGenerationSchema(type: Double.self)
        case .boolean: return DynamicGenerationSchema(type: Bool.self)
        }
    }

    static func value(of type: NFKToolParameterType,
                      from content: GeneratedContent,
                      property: String) throws -> Any {
        switch type {
        case .string:  return try content.value(String.self, forProperty: property)
        case .integer: return try content.value(Int.self, forProperty: property)
        case .number:  return try content.value(Double.self, forProperty: property)
        case .boolean: return try content.value(Bool.self, forProperty: property)
        }
    }
}
