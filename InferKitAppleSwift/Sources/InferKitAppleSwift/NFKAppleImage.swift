//
//  NFKAppleImage.swift
//  InferKitAppleSwift
//

import CoreGraphics
import Foundation
import InferKit

/// The image reading and the error shape the package's Vision backends share.
enum NFKAppleImage {

    /// The request's image for a key, converted by the core's own coder so every engine in the
    /// toolkit accepts the same inputs.
    static func cgImage(in request: NFKInferenceRequest, key: String) -> CGImage? {
        guard let value = request.input(forKey: key) else {
            return nil
        }
        return NFKImageCoding.cgImage(forImage: value)
    }

    static func error(_ code: NFKInferenceError, _ reason: String) -> NSError {
        NSError(domain: NFKInferenceErrorDomain,
                code: code.rawValue,
                userInfo: [NSLocalizedDescriptionKey: reason])
    }
}
