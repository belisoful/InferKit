//
//  NFKMLXCatalogVideo.swift
//  InferKitMLX
//
//  The catalog entries of the video models: video generation, frame interpolation, video super-resolution,
//  video matting, and video understanding.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The catalog entries of the video models, and the registrations their registry names need.
enum NFKMLXCatalogVideo {

    /// The video models' entries, in the order ``NFKMLXModelDescriptors`` lists them.
    static var entries: [NFKMLXCatalogEntry] {
        []
    }

    /// Registers the video models that have a factory and no registry name, under the names their entries
    /// give. ``NFKMLXReferenceModels/registerAll()`` calls it.
    static func registerAdditionalModels() {
    }
}
