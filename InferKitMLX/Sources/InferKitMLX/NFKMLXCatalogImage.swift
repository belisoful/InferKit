//
//  NFKMLXCatalogImage.swift
//  InferKitMLX
//
//  The catalog entries of the image models: image restoration, upscaling, depth, segmentation, matting,
//  detection, pose, embedding, and image generation.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

/// The catalog entries of the image models, and the registrations their registry names need.
enum NFKMLXCatalogImage {

    /// The image models' entries, in the order ``NFKMLXModelDescriptors`` lists them.
    static var entries: [NFKMLXCatalogEntry] {
        []
    }

    /// Registers the image models that have a factory and no registry name, under the names their entries
    /// give. ``NFKMLXReferenceModels/registerAll()`` calls it.
    static func registerAdditionalModels() {
    }
}
