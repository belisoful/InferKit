//
//  NFKMLXCatalogAudio.swift
//  InferKitMLX
//
//  The catalog entries of the audio models: speech recognition, speech synthesis, speech enhancement, source
//  separation, voice activity, codecs, and music.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The catalog entries of the audio models, and the registrations their registry names need.
enum NFKMLXCatalogAudio {

    /// The audio models' entries, in the order ``NFKMLXModelDescriptors`` lists them.
    static var entries: [NFKMLXCatalogEntry] {
        []
    }

    /// Registers the audio models that have a factory and no registry name, under the names their entries
    /// give. ``NFKMLXReferenceModels/registerAll()`` calls it.
    static func registerAdditionalModels() {
    }
}
