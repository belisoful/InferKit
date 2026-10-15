//
//  NFKMLXCatalogLanguage.swift
//  InferKitMLX
//
//  The catalog entries of the language models: text generation, vision-language, translation, embedding,
//  reranking, forecasting, and typed decisions.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The catalog entries of the language models, and the registrations their registry names need.
enum NFKMLXCatalogLanguage {

    /// The language models' entries, in the order ``NFKMLXModelDescriptors`` lists them.
    static var entries: [NFKMLXCatalogEntry] {
        []
    }

    /// Registers the language models that have a factory and no registry name, under the names their entries
    /// give. ``NFKMLXReferenceModels/registerAll()`` calls it.
    static func registerAdditionalModels() {
    }
}
