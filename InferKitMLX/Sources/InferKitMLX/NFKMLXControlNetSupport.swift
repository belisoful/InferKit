//
//  NFKMLXControlNetSupport.swift
//  InferKitMLX
//
//  What the FLUX.1 and SD3 ControlNet backends share: the request keys a ControlNet reads beyond the
//  core ones, and the control image as the pipelines take it.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// The request parameters a ControlNet backend reads beyond the core keys.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXControlNetParameterKey)
public final class NFKMLXControlNetParameterKey: NSObject {
    /// The weight the ControlNet's residuals carry, as an `NSNumber` (default 1, the reference
    /// pipelines' `controlnet_conditioning_scale`). Introduced in InferKit 0.4.0.
    @objc public static let conditioningScale = "NFKMLXParameterControlNetConditioningScale"
    /// The control type a union ControlNet reads, as an `NSNumber` index in the release's own numbering,
    /// which its model card lists (`Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro`: 0 canny, 1 tile,
    /// 2 depth, 3 blur, 4 pose, 5 gray, 6 low quality). The FLUX.1 ControlNet backend requires it for a
    /// union ControlNet and refuses it for a single-type one; the SD3 ControlNet backend does not read it.
    /// Introduced in InferKit 0.4.0.
    @objc public static let controlMode = "NFKMLXParameterControlNetMode"
}

/// The control image as the ControlNet pipelines take it.
enum NFKMLXControlImage {

    /// `value` (a `CGImage` or a texture) resized to `width` × `height`, as `[1, height, width, 3]` in
    /// `[-1, 1]`, which is how the reference pipelines' image processor hands it to the autoencoder. An
    /// image with alpha has it dropped and its stored color kept, as the processor's `convert_to_rgb` does.
    static func pixels(_ value: Any, width: Int, height: Int) throws -> MLXArray {
        let image = NFKMLXImageBridge.droppingAlpha(try NFKMLXGenerationBackend.cgImage(fromInput: value))
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = bytes.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: width, height: height,
                                          bitsPerComponent: 8, bytesPerRow: width * 4,
                                          space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else {
                return false
            }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw NFKMLXError.unsupportedInput }
        let tensor = NFKMLXImageBridge.tensor(rgba: bytes, width: width, height: height, channels: 3)
        return (tensor * 2 - 1).expandedDimensions(axis: 0)
    }

    /// The output size a request names, or the control image's own size rounded down to `multiple`.
    static func size(_ request: NFKInferenceRequest, control value: Any, multiple: Int) throws -> (width: Int, height: Int) {
        let image = try NFKMLXGenerationBackend.cgImage(fromInput: value)
        let width = NFKMLXGenerationBackend.number(request, NFKParameterWidth)?.intValue
            ?? image.width / multiple * multiple
        let height = NFKMLXGenerationBackend.number(request, NFKParameterHeight)?.intValue
            ?? image.height / multiple * multiple
        guard width > 0, height > 0, width % multiple == 0, height % multiple == 0 else {
            throw NFKMLXError.unsupportedConfiguration("an image's sides are positive multiples of \(multiple)")
        }
        return (width, height)
    }
}
