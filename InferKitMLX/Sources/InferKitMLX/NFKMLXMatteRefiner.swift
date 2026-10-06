//
//  NFKMLXMatteRefiner.swift
//  InferKitMLX
//

import CoreGraphics
import Foundation
import MLX

/// A matting model's matte finished by ``NFKMLXMatteOperations``, reachable from Objective-C. Introduced
/// in InferKit 0.4.0.
///
/// @discussion Every setting starts at its neutral value, so a new refiner returns the matte it is
/// given. The steps run in this order, each only when its setting departs from neutral:
///
/// 1. Clip to `clipBlack`...`clipWhite`, sparing the plate's edges when `edgeProtectionRadius` is set.
/// 2. Gamma.
/// 3. Speck removal, then hole filling.
/// 4. The guided filter against the plate.
/// 5. Growth or shrinkage, then the feather.
/// 6. The garbage and core mattes.
///
/// The images go through the matting backend's 8-bit bridge, read and written in device RGB, which the
/// compositing steps treat as sRGB. Swift code holding float arrays calls ``refined(_:plate:garbage:core:)``
/// and the ``NFKMLXMatteOperations`` functions directly.
@objc(NFKMLXMatteRefiner)
public final class NFKMLXMatteRefiner: NSObject {
    /// The matte value that becomes 0 (default 0).
    @objc public var clipBlack: Float = 0
    /// The matte value that becomes 1 (default 1).
    @objc public var clipWhite: Float = 1
    /// The radius in pixels around the plate's edges the clip leaves alone (default 0, clip everywhere).
    @objc public var edgeProtectionRadius: Int = 0
    /// The color change within that radius that marks an edge (default 0.1).
    @objc public var edgeTolerance: Float = 0.1
    /// The gamma applied after the clip; above 1 lifts partial values (default 1).
    @objc public var gamma: Float = 1
    /// Islands above half holding fewer pixels than this are removed (default 0, off).
    @objc public var minimumSpeckArea: Int = 0
    /// Holes at or below half holding at most this many pixels, away from the image edge, are filled
    /// (default 0, off).
    @objc public var maximumHoleArea: Int = 0
    /// The guided filter's radius in pixels (default 0, off).
    @objc public var guidedFilterRadius: Int = 0
    /// The guided filter's regularizer; smaller follows the plate's edges more closely (default 1e-4).
    @objc public var guidedFilterEpsilon: Float = 1e-4
    /// The scale divisor the guided filter computes its coefficients at (default 4).
    @objc public var guidedFilterSubsampling: Int = 4
    /// Growth (positive) or shrinkage (negative) in pixels by a disc (default 0).
    @objc public var morphologyRadius: Float = 0
    /// The feather's reach in pixels (default 0, off).
    @objc public var featherRadius: Float = 0

    /// The refined matte `[H, W, 1]` from a matte `[H, W, 1]` and its plate `[H, W, 3]`, with optional
    /// garbage and core mattes `[H, W, 1]`.
    public func refined(_ alpha: MLXArray, plate: MLXArray, garbage: MLXArray? = nil, core: MLXArray? = nil) -> MLXArray {
        var result = alpha
        if clipBlack > 0 || clipWhite < 1 {
            let protected = edgeProtectionRadius > 0
                ? NFKMLXMatteOperations.edges(of: plate, radius: edgeProtectionRadius, tolerance: edgeTolerance) : nil
            result = NFKMLXMatteOperations.clipped(result, black: clipBlack, white: clipWhite, protected: protected)
        }
        if gamma != 1 {
            result = NFKMLXMatteOperations.gammaAdjusted(result, gamma: gamma)
        }
        if minimumSpeckArea > 0 {
            result = NFKMLXMatteOperations.specksRemoved(result, minimumArea: minimumSpeckArea)
        }
        if maximumHoleArea > 0 {
            result = NFKMLXMatteOperations.holesFilled(result, maximumArea: maximumHoleArea)
        }
        if guidedFilterRadius > 0 {
            result = NFKMLXMatteOperations.guidedFiltered(result, guide: plate, radius: guidedFilterRadius,
                                                          epsilon: guidedFilterEpsilon,
                                                          subsampling: guidedFilterSubsampling)
        }
        if morphologyRadius != 0 {
            result = NFKMLXMatteOperations.morphed(result, radius: morphologyRadius)
        }
        if featherRadius > 0 {
            result = NFKMLXMatteOperations.feathered(result, radius: featherRadius)
        }
        return NFKMLXMatteOperations.masked(result, garbage: garbage, core: core)
    }

    /// Refines a matte (a gray `CGImage` or `MTLTexture`, its first channel read) against its plate, with
    /// optional garbage and core mattes of the same size, and returns the refined matte as a gray
    /// `CGImage`.
    @objc(refineMatte:plate:garbageMatte:coreMatte:error:)
    public func refine(matte: Any, plate: Any, garbageMatte: Any?, coreMatte: Any?) throws -> CGImage {
        let space = CGColorSpaceCreateDeviceRGB()
        let alpha = try NFKMLXImageBridge.tensor(from: matte, channels: 1, colorSpace: space)
        let image = try NFKMLXImageBridge.tensor(from: plate, channels: 3, colorSpace: space)
        let garbage = try garbageMatte.map { try NFKMLXImageBridge.tensor(from: $0, channels: 1, colorSpace: space) }
        let core = try coreMatte.map { try NFKMLXImageBridge.tensor(from: $0, channels: 1, colorSpace: space) }
        for other in [image, garbage, core].compactMap({ $0 }) where other.dim(0) != alpha.dim(0) || other.dim(1) != alpha.dim(1) {
            throw NFKMLXError.unsupportedInput
        }
        return try NFKMLXImageBridge.cgImage(from: refined(alpha, plate: image, garbage: garbage, core: core),
                                             options: NFKMLXImageOptions(colorSpace: space))
    }

    /// The foreground and background colors that composite to the plate under the matte, the
    /// foreground returned as a straight RGBA `CGImage` carrying the matte: the edge decontamination of
    /// ``NFKMLXMatteOperations/estimatedColors(image:alpha:omega:epsilon:smallIterations:largeIterations:smallSize:)``,
    /// run in linear light.
    @objc(decontaminatedForegroundForPlate:matte:error:)
    public func decontaminatedForeground(plate: Any, matte: Any) throws -> CGImage {
        let space = CGColorSpaceCreateDeviceRGB()
        let image = try NFKMLXImageBridge.tensor(from: plate, channels: 3, colorSpace: space)
        let alpha = try NFKMLXImageBridge.tensor(from: matte, channels: 1, colorSpace: space)
        let colors = NFKMLXMatteOperations.estimatedColors(image: Self.linearized(image), alpha: alpha)
        return try NFKMLXImageBridge.cgImage(from: concatenated([Self.delinearized(colors.foreground), alpha], axis: -1),
                                             options: NFKMLXImageOptions(colorSpace: space))
    }

    /// A straight foreground image under its matte over a background, composited in linear light with a
    /// light wrap of `radius` pixels at `strength` (0 for none), returned as an opaque `CGImage`.
    @objc(compositeForeground:matte:overBackground:lightWrapRadius:lightWrapStrength:error:)
    public func composite(foreground: Any, matte: Any, background: Any, lightWrapRadius: Float,
                          lightWrapStrength: Float) throws -> CGImage {
        let space = CGColorSpaceCreateDeviceRGB()
        let front = Self.linearized(try NFKMLXImageBridge.tensor(from: foreground, channels: 3, colorSpace: space))
        let alpha = try NFKMLXImageBridge.tensor(from: matte, channels: 1, colorSpace: space)
        let back = Self.linearized(try NFKMLXImageBridge.tensor(from: background, channels: 3, colorSpace: space))
        let composite = NFKMLXMatteOperations.lightWrapped(foreground: front, alpha: alpha, background: back,
                                                           radius: lightWrapRadius, strength: lightWrapStrength)
        return try NFKMLXImageBridge.cgImage(from: Self.delinearized(composite), options: NFKMLXImageOptions(colorSpace: space))
    }

    /// The piecewise sRGB transfer function inverted.
    static func linearized(_ srgb: MLXArray) -> MLXArray {
        let x = maximum(srgb, 0)
        return which(x .<= 0.04045, x / 12.92, pow((x + 0.055) / 1.055, 2.4))
    }

    /// The piecewise sRGB transfer function.
    static func delinearized(_ linear: MLXArray) -> MLXArray {
        let x = maximum(linear, 0)
        return which(x .<= 0.0031308, x * 12.92, 1.055 * pow(x, Float(1.0 / 2.4)) - 0.055)
    }
}

/// A moving average of a clip's mattes that stops where the picture moves, reachable from Objective-C.
/// Introduced in InferKit 0.4.0.
///
/// @discussion Each frame's matte moves `weight` of the way toward the previous output where the
/// plate's largest channel change from the previous plate is below `motionThreshold`, and keeps its own
/// value elsewhere. The first frame, and any frame whose size differs from the last, passes unchanged.
@objc(NFKMLXMatteTemporalBlender)
public final class NFKMLXMatteTemporalBlender: NSObject {
    /// How far a still pixel moves toward the previous output, 0...1 (default 0.5).
    @objc public var weight: Float = 0.5
    /// The plate change below which a pixel counts as still (default 0.05).
    @objc public var motionThreshold: Float = 0.05

    private var previousMatte: MLXArray?
    private var previousPlate: MLXArray?
    private let lock = NSLock()

    /// The blended matte `[H, W, 1]` for this frame's matte and plate `[H, W, 3]`.
    public func blended(_ alpha: MLXArray, plate: MLXArray) -> MLXArray {
        lock.lock()
        defer { lock.unlock() }
        var result = alpha
        if let previousMatte, let previousPlate, previousMatte.shape == alpha.shape, previousPlate.shape == plate.shape {
            result = NFKMLXMatteOperations.temporallyBlended(alpha, previous: previousMatte, image: plate,
                                                             previousImage: previousPlate, weight: weight,
                                                             motionThreshold: motionThreshold)
        }
        eval(result, plate)
        previousMatte = result
        previousPlate = plate
        return result
    }

    /// Blends one frame's matte (a gray `CGImage` or `MTLTexture`) given its plate, and returns the
    /// blended matte as a gray `CGImage`.
    @objc(blendMatte:plate:error:)
    public func blend(matte: Any, plate: Any) throws -> CGImage {
        let space = CGColorSpaceCreateDeviceRGB()
        let alpha = try NFKMLXImageBridge.tensor(from: matte, channels: 1, colorSpace: space)
        let image = try NFKMLXImageBridge.tensor(from: plate, channels: 3, colorSpace: space)
        return try NFKMLXImageBridge.cgImage(from: blended(alpha, plate: image), options: NFKMLXImageOptions(colorSpace: space))
    }

    /// Forgets the previous frame, as at a cut.
    @objc public func reset() {
        lock.lock()
        defer { lock.unlock() }
        previousMatte = nil
        previousPlate = nil
    }
}
