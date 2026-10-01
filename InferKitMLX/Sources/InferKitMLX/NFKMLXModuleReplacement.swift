//
//  NFKMLXModuleReplacement.swift
//  InferKitMLX
//
//  Putting a module in place of another inside a model, at any position MLX can hold one.
//
//  MLX's `update(modules:)` reads a container from what the update names, not from the module it is
//  updating. For an array it branches on the array's FIRST named entry: a module there replaces the
//  whole array with the entries named, so naming a prefix drops the tail with no error; an empty first
//  entry is refused outright. A dictionary of modules is replaced by the keys named, so naming one key
//  drops the rest. A path into `layers.1` alone, or an adapter set that skips a container's last
//  entries, therefore either fails or truncates the model.
//
//  Updating the module that owns each replaced child addresses no container at all. A child held
//  directly in an array or a dictionary goes back as its owner's whole container, every other entry
//  unchanged.
//

import Foundation
import MLX
import MLXNN

enum NFKMLXModuleReplacement {

    /// Puts each `(path, module)` in place in `model`, one at a time through its owner.
    static func place(_ replacements: [(String, Module)], in model: Module) throws {
        for (path, module) in replacements {
            try place(module, at: path, in: model)
        }
    }

    /// Puts `module` at `path` in `model` through the module that owns it.
    ///
    /// - Throws: MLX's own error when the owner cannot take the child (a plain property, for one), and
    ///   `NFKMLXError.unsupportedConfiguration` when no module owns the path.
    static func place(_ module: Module, at path: String, in model: Module) throws {
        let components = path.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        let owners = Dictionary(model.namedModules(), uniquingKeysWith: { first, _ in first })
        func owner(_ prefix: ArraySlice<String>) -> Module? {
            prefix.isEmpty ? model : owners[prefix.joined(separator: ".")]
        }
        guard let key = components.last, !key.isEmpty else {
            throw NFKMLXError.unsupportedConfiguration("an empty path names no child")
        }
        if Int(key) == nil, let parent = owner(components.dropLast()) {
            try parent.update(modules: ModuleChildren.unflattened([(key, module)]), verify: .none)
            return
        }
        guard components.count >= 2, let parent = owner(components.dropLast(2)) else {
            throw NFKMLXError.unsupportedConfiguration("no module owns \(path)")
        }
        let containerKey = components[components.count - 2]
        var entries = [(String, Module)]()
        switch parent.children()[containerKey] {
        case .array(let items)?:
            guard let index = Int(key), items.indices.contains(index) else {
                throw NFKMLXError.unsupportedConfiguration("\(path) is past the end of its array")
            }
            for (position, item) in items.enumerated() {
                guard case .value(let existing) = item else {
                    throw NFKMLXError.unsupportedConfiguration("\(containerKey) in the owner of \(path) nests a container")
                }
                entries.append(("\(containerKey).\(position)", position == index ? module : existing))
            }
        case .dictionary(let items)?:
            guard items[key] != nil else {
                throw NFKMLXError.unsupportedConfiguration("\(path) names no entry of its dictionary")
            }
            for (name, item) in items {
                guard case .value(let existing) = item else {
                    throw NFKMLXError.unsupportedConfiguration("\(containerKey) in the owner of \(path) nests a container")
                }
                entries.append(("\(containerKey).\(name)", name == key ? module : existing))
            }
        default:
            throw NFKMLXError.unsupportedConfiguration("no module owns \(path)")
        }
        try parent.update(modules: ModuleChildren.unflattened(entries), verify: .none)
    }
}
