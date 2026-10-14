//
//  NFKMLXDepth3EstimatorBackend.swift
//  InferKitMLX
//
//  The Depth Anything 3 estimator behind the InferKit contract: the depth image, the camera, and the
//  ray map from one forward pass.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

extension NFKMLXDepth3Estimator {

    /// The identifier the estimator's backend reports. Introduced in InferKit 0.4.0.
    @objc public static let estimatorModelName = "depth-anything-3-estimator"

    /// The estimator behind the InferKit contract.
    ///
    /// @discussion The backend reads `NFKInputImage` (required; a `CGImage` or an `MTLTexture`) and runs
    /// the model once. The result carries:
    /// - `NFKOutputImage`: the depth map as a grayscale `CGImage` at the input's size, as
    ///   ``NFKMLXDepthAnything3/backend(variant:weightsURL:)`` returns it.
    /// - `NFKOutputStructured`: an `NSDictionary` with three entries.
    ///   - `camera`: `translation` (three numbers) and `rotation` (nine numbers, row-major) of the
    ///     camera-to-world pose, `focalLengthX` and `focalLengthY` in pixels at the input's size, and
    ///     `fieldOfViewX` and `fieldOfViewY` in radians, as ``camera(for:)`` reports them.
    ///   - `rays`: the six Plücker channels per position at the head's resolution, as `width`,
    ///     `height`, `channels` (6), and `data`, an `NSData` of float32 values in row-major
    ///     `[height, width, channels]` order.
    ///   - `rayConfidence`: the ray branch's confidence in the same layout with `channels` 1.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(inferenceBackend)
    public func inferenceBackend() -> any NFKInferenceBackend {
        NFKMLXGenerationBackend(identifier: Self.estimatorModelName, inputKeys: [NFKInputImage],
                                parameterKeys: []) { [self] request, job in
            guard let image = try NFKMLXGenerationBackend.image(request, NFKInputImage) else {
                throw NFKMLXError.unsupportedInput
            }
            job.reportProgress(0)
            let tensor = try NFKMLXImageBridge.tensor(from: image, channels: 3,
                                                      colorSpace: CGColorSpaceCreateDeviceRGB())
            let (prediction, pose) = net.predict(net.prepared(tensor))
            let depth = net.grayscaleDepth(prediction.depth, height: image.height, width: image.width)
            eval(depth, prediction.ray, prediction.rayConfidence)
            if job.status == .cancelled { return nil }
            let camera = NFKMLXDepth3Camera.camera(from: pose, imageSize: (image.height, image.width))
            let structured: [String: Any] = [
                "camera": [
                    "translation": camera.translation,
                    "rotation": camera.rotation,
                    "focalLengthX": camera.focalLengthX,
                    "focalLengthY": camera.focalLengthY,
                    "fieldOfViewX": camera.fieldOfViewX,
                    "fieldOfViewY": camera.fieldOfViewY,
                ] as [String: Any],
                "rays": Self.map(prediction.ray),
                "rayConfidence": Self.map(prediction.rayConfidence.expandedDimensions(axis: -1)),
            ]
            let result = NFKInferenceResult(outputs: [NFKOutputImage: try NFKMLXGenerationBackend.cgImage(depth),
                                                      NFKOutputStructured: structured])
            job.reportProgress(1)
            return result
        }
    }

    /// `[1, height, width, channels]` as the dictionary the backend reports a map in.
    static func map(_ array: MLXArray) -> [String: Any] {
        let values = array.asType(.float32).asArray(Float.self)
        return ["width": array.dim(2), "height": array.dim(1), "channels": array.dim(3),
                "data": values.withUnsafeBufferPointer { Data(buffer: $0) }]
    }
}

extension NFKMLXDepthAnything3Net {

    /// The head's depth `[1, inputSize, inputSize]` resized to `height` × `width` and normalized to a
    /// grayscale image `[height, width, 3]`, which is what ``depth(_:)`` returns for the same image.
    func grayscaleDepth(_ depthMap: MLXArray, height: Int, width: Int) -> MLXArray {
        let full = NFKDA3Resample.bilinearAlignCorners(
            depthMap.reshaped([1, configuration.inputSize, configuration.inputSize, 1]),
            height: height, width: width)
        let minimum = full.min()
        let span = maximum(full.max() - minimum, MLXArray(1e-6))
        let normalized = ((full - minimum) / span).reshaped([height, width, 1])
        return concatenated([normalized, normalized, normalized], axis: 2)
    }
}
