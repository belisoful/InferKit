//
//  NFKSchema.swift
//  InferKitFoundationModels
//

import Foundation
import FoundationModels

/// A JSON Schema keyword the framework's schema cannot express, named by its path.
public struct NFKSchemaError: Error, CustomStringConvertible, Equatable {
    public let path: String
    public let reason: String
    public var description: String { "\(path): \(reason)" }
}

/// Builds Foundation Models schemas from JSON Schema dictionaries (the core's `NFKParameterJSONSchema`
/// and `NFKParameterTools` shapes) and reads generated content back out as JSON objects. Shared by
/// tool arguments (`NFKToolAdapter`) and structured output.
///
/// The supported subset: `type` object (`properties`, `required`), array (`items`, `minItems`,
/// `maxItems`), string (`pattern`, `const`, `enum`), integer and number (`minimum`, `maximum`),
/// boolean, `anyOf` / `oneOf` of schemas, `$ref` to `#/$defs/…` or `#/definitions/…`, and
/// `description` on any of them. A `type` list takes its first non-null entry. Anything else throws
/// `NFKSchemaError` naming the path, since a keyword silently dropped would change what the model is
/// allowed to produce.
enum NFKSchema {

    /// A `GenerationSchema` for a JSON Schema object. `name` names the root; `$defs` / `definitions`
    /// become the schema's dependencies under their own keys.
    static func generationSchema(name: String, json: [String: Any]) throws -> GenerationSchema {
        let root = try dynamicSchema(name: name, json: json, path: name)
        var dependencies: [DynamicGenerationSchema] = []
        for key in ["$defs", "definitions"] {
            guard let definitions = json[key] as? [String: Any] else { continue }
            for (definitionName, value) in definitions.sorted(by: { $0.key < $1.key }) {
                guard let definition = value as? [String: Any] else {
                    throw NFKSchemaError(path: "\(key).\(definitionName)", reason: "a definition is a schema object")
                }
                dependencies.append(try dynamicSchema(name: definitionName, json: definition, path: definitionName))
            }
        }
        return try GenerationSchema(root: root, dependencies: dependencies)
    }

    /// A schema whose value is exactly one of the strings (the core's `NFKParameterChoices`).
    static func choiceSchema(_ choices: [String]) throws -> GenerationSchema {
        let root = DynamicGenerationSchema(name: "Choice", description: "exactly one of the allowed values", anyOf: choices)
        return try GenerationSchema(root: root, dependencies: [])
    }

    static func dynamicSchema(name: String, json: [String: Any], path: String) throws -> DynamicGenerationSchema {
        let description = json["description"] as? String

        if let reference = json["$ref"] as? String {
            guard let referenced = reference.split(separator: "/").last, reference.hasPrefix("#/") else {
                throw NFKSchemaError(path: path, reason: "only local references (#/$defs/Name) are supported")
            }
            return DynamicGenerationSchema(referenceTo: String(referenced))
        }
        if let choices = json["enum"] as? [Any] {
            guard let strings = choices as? [String], !strings.isEmpty else {
                throw NFKSchemaError(path: path, reason: "enum values are non-empty strings")
            }
            return DynamicGenerationSchema(name: name, description: description, anyOf: strings)
        }
        if let alternatives = (json["anyOf"] ?? json["oneOf"]) as? [Any] {
            let schemas = try alternatives.enumerated().map { index, value -> DynamicGenerationSchema in
                guard let alternative = value as? [String: Any] else {
                    throw NFKSchemaError(path: "\(path).anyOf[\(index)]", reason: "an alternative is a schema object")
                }
                return try dynamicSchema(name: "\(name)Option\(index)", json: alternative, path: "\(path).anyOf[\(index)]")
            }
            return DynamicGenerationSchema(name: name, description: description, anyOf: schemas)
        }

        switch try typeName(of: json, path: path) {
        case "object":
            return try objectSchema(name: name, description: description, json: json, path: path)
        case "array":
            guard let items = json["items"] as? [String: Any] else {
                throw NFKSchemaError(path: path, reason: "an array schema names its items")
            }
            let element = try dynamicSchema(name: "\(name)Item", json: items, path: "\(path).items")
            return DynamicGenerationSchema(arrayOf: element,
                                           minimumElements: json["minItems"] as? Int,
                                           maximumElements: json["maxItems"] as? Int)
        case "string":
            return try stringSchema(json: json, path: path)
        case "integer":
            return DynamicGenerationSchema(type: Int.self, guides: integerGuides(json))
        case "number":
            return DynamicGenerationSchema(type: Double.self, guides: numberGuides(json))
        case "boolean":
            return DynamicGenerationSchema(type: Bool.self)
        case let other:
            throw NFKSchemaError(path: path, reason: "unsupported type \"\(other)\"")
        }
    }

    // MARK: Reading results

    /// The generated content as a Foundation JSON object (dictionary, array, string, number, bool).
    static func jsonObject(from content: GeneratedContent) -> Any? {
        let data = Data(content.jsonString.utf8)
        return try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
    }

    /// The generated content as a JSON object keyed by property name, or empty when it is not an object.
    static func dictionary(from content: GeneratedContent) -> [String: Any] {
        jsonObject(from: content) as? [String: Any] ?? [:]
    }

    // MARK: Pieces

    private static func typeName(of json: [String: Any], path: String) throws -> String {
        if let type = json["type"] as? String {
            return type
        }
        if let types = json["type"] as? [String], let first = types.first(where: { $0 != "null" }) {
            return first
        }
        if json["properties"] != nil {
            return "object"
        }
        throw NFKSchemaError(path: path, reason: "a schema names its type")
    }

    private static func objectSchema(name: String, description: String?,
                                     json: [String: Any], path: String) throws -> DynamicGenerationSchema {
        let properties = json["properties"] as? [String: Any] ?? [:]
        let required = Set(json["required"] as? [String] ?? [])
        // JSON Schema property order is not significant, but the model reads the schema in order, so
        // a stable order keeps a prompt reproducible.
        let ordered = properties.keys.sorted()
        let dynamicProperties = try ordered.map { propertyName -> DynamicGenerationSchema.Property in
            guard let property = properties[propertyName] as? [String: Any] else {
                throw NFKSchemaError(path: "\(path).\(propertyName)", reason: "a property is a schema object")
            }
            let schema = try dynamicSchema(name: "\(name).\(propertyName)", json: property, path: "\(path).\(propertyName)")
            return DynamicGenerationSchema.Property(name: propertyName,
                                                    description: property["description"] as? String,
                                                    schema: schema,
                                                    isOptional: !required.contains(propertyName))
        }
        return DynamicGenerationSchema(name: name, description: description, properties: dynamicProperties)
    }

    private static func stringSchema(json: [String: Any], path: String) throws -> DynamicGenerationSchema {
        var guides: [GenerationGuide<String>] = []
        if let constant = json["const"] as? String {
            guides.append(.constant(constant))
        }
        if let pattern = json["pattern"] as? String {
            guard let regex = try? Regex(pattern) else {
                throw NFKSchemaError(path: path, reason: "pattern is not a valid regular expression")
            }
            guides.append(.pattern(regex))
        }
        return DynamicGenerationSchema(type: String.self, guides: guides)
    }

    private static func integerGuides(_ json: [String: Any]) -> [GenerationGuide<Int>] {
        let minimum = (json["minimum"] as? NSNumber)?.intValue
        let maximum = (json["maximum"] as? NSNumber)?.intValue
        switch (minimum, maximum) {
        case let (low?, high?): return [.range(low...high)]
        case let (low?, nil):   return [.minimum(low)]
        case let (nil, high?):  return [.maximum(high)]
        case (nil, nil):        return []
        }
    }

    private static func numberGuides(_ json: [String: Any]) -> [GenerationGuide<Double>] {
        let minimum = (json["minimum"] as? NSNumber)?.doubleValue
        let maximum = (json["maximum"] as? NSNumber)?.doubleValue
        switch (minimum, maximum) {
        case let (low?, high?): return [.range(low...high)]
        case let (low?, nil):   return [.minimum(low)]
        case let (nil, high?):  return [.maximum(high)]
        case (nil, nil):        return []
        }
    }
}
