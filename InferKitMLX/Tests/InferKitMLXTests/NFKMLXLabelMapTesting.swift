//
//  NFKMLXLabelMapTesting.swift
//  InferKitMLXTests
//
//  Reading the grayscale label maps the segmentation backends emit.
//

import CoreGraphics
import Foundation
@testable import InferKitMLX

enum NFKMLXLabelMapTesting {

    /// A mid-gray RGBA image.
    static func solid(_ width: Int, _ height: Int) -> CGImage {
        let pixels = [UInt8](repeating: 128, count: width * height * 4)
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    /// The distinct gray levels of a label map a backend returned.
    static func levels(_ value: Any?) throws -> Set<UInt8> {
        guard let value, CFGetTypeID(value as CFTypeRef) == CGImage.typeID else {
            throw NFKMLXError.noOutput
        }
        let image = value as! CGImage
        var pixels = [UInt8](repeating: 0, count: image.width * image.height)
        let context = CGContext(data: &pixels, width: image.width, height: image.height, bitsPerComponent: 8,
                                bytesPerRow: image.width, space: CGColorSpaceCreateDeviceGray(),
                                bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return Set(pixels)
    }
}
