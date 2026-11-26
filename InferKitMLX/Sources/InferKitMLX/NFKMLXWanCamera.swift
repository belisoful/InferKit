//
//  NFKMLXWanCamera.swift
//  InferKitMLX
//
//  A camera trajectory as the Wan 2.2 Fun Control-Camera releases read it: CameraCtrl's pose file (one
//  line per frame: an index, the normalized intrinsics fx fy cx cy, two unused values, and the 3 × 4
//  world-to-camera matrix) turned into a per-pixel Plücker embedding of each frame's rays, as VideoX-Fun's
//  `process_pose_file` and `ray_condition` compute it.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX

/// A camera trajectory in CameraCtrl's pose format, and the Plücker embedding a Wan 2.2 Fun Control-Camera
/// release reads from it.
///
/// @discussion Every pose is made relative to the first (`get_relative_pose`), the intrinsics are rescaled
/// from the poses' own aspect (1280 × 720 unless given) to the clip's, and each pixel's ray, normalized,
/// becomes six channels: the ray origin crossed with its direction, then the direction. Introduced in
/// InferKit 0.4.0.
@objc(NFKMLXWanCameraTrajectory)
public final class NFKMLXWanCameraTrajectory: NSObject {
    /// One frame's camera.
    struct Pose {
        var fx: Double, fy: Double, cx: Double, cy: Double
        /// The world-to-camera matrix, row-major 4 × 4.
        var worldToCamera: [Double]
    }

    let poses: [Pose]

    /// The number of frames the trajectory describes.
    @objc public var frameCount: Int { poses.count }

    /// Reads a pose file's text: a header line, then one line per frame of 19 numbers.
    @objc public init(poseFileContents text: String) throws {
        var poses = [Pose]()
        for line in text.split(whereSeparator: \.isNewline).dropFirst() {
            let values = line.split(separator: " ").compactMap { Double($0) }
            guard values.count >= 19 else {
                throw NFKMLXError.unsupportedConfiguration("a camera pose line holds \(values.count) numbers, not 19")
            }
            poses.append(Pose(fx: values[1], fy: values[2], cx: values[3], cy: values[4],
                              worldToCamera: Array(values[7 ..< 19]) + [0, 0, 0, 1]))
        }
        guard !poses.isEmpty else {
            throw NFKMLXError.unsupportedConfiguration("the camera pose file holds no frames")
        }
        self.poses = poses
    }

    /// Reads a pose file.
    @objc public convenience init(poseFileURL url: URL) throws {
        try self.init(poseFileContents: try String(contentsOf: url, encoding: .utf8))
    }

    /// The Plücker embedding `[6, frames, height, width]` of the first `frames` poses at a clip's size.
    ///
    /// @discussion `poseWidth` and `poseHeight` are the size the poses were recorded at, whose aspect the
    /// intrinsics are rescaled from; VideoX-Fun's default is 1280 × 720.
    public func plucker(width: Int, height: Int, frames: Int? = nil, poseWidth: Double = 1280,
                        poseHeight: Double = 720) throws -> MLXArray {
        let count = frames ?? poses.count
        guard count <= poses.count else {
            throw NFKMLXError.unsupportedConfiguration("the trajectory has \(poses.count) frames, fewer than \(count)")
        }
        let sampleRatio = Double(width) / Double(height), poseRatio = poseWidth / poseHeight
        let relative = Self.relativeCameraToWorld(poses)
        var values = [Float](repeating: 0, count: 6 * count * height * width)
        let plane = height * width
        for frame in 0 ..< count {
            var pose = poses[frame]
            if poseRatio > sampleRatio {
                pose.fx = Double(height) * poseRatio * pose.fx / Double(width)
            } else {
                pose.fy = Double(width) / poseRatio * pose.fy / Double(height)
            }
            let fx = Float(pose.fx * Double(width)), fy = Float(pose.fy * Double(height))
            let cx = Float(pose.cx * Double(width)), cy = Float(pose.cy * Double(height))
            let c2w = relative[frame]
            let origin = [c2w[3], c2w[7], c2w[11]]
            for y in 0 ..< height {
                for x in 0 ..< width {
                    let xs = (Float(x) + 0.5 - cx) / fx, ys = (Float(y) + 0.5 - cy) / fy
                    let norm = (xs * xs + ys * ys + 1).squareRoot()
                    let d = [xs / norm, ys / norm, 1 / norm]
                    // `directions @ c2w[:3, :3]ᵀ`: each component is a row of the rotation against the direction.
                    let ray = (0 ..< 3).map { row in d[0] * c2w[row * 4] + d[1] * c2w[row * 4 + 1] + d[2] * c2w[row * 4 + 2] }
                    let moment = [origin[1] * ray[2] - origin[2] * ray[1], origin[2] * ray[0] - origin[0] * ray[2],
                                  origin[0] * ray[1] - origin[1] * ray[0]]
                    let pixel = y * width + x
                    for channel in 0 ..< 3 {
                        values[(channel * count + frame) * plane + pixel] = moment[channel]
                        values[((channel + 3) * count + frame) * plane + pixel] = ray[channel]
                    }
                }
            }
        }
        return MLXArray(values).reshaped([6, count, height, width])
    }

    /// Each frame's camera-to-world matrix relative to the first camera, rounded to float32 as the
    /// reference's `get_relative_pose` stores it: the first is the identity, every other the first's
    /// world-to-camera times its own camera-to-world.
    static func relativeCameraToWorld(_ poses: [Pose]) -> [[Float]] {
        let first = poses[0].worldToCamera
        var result: [[Float]] = [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
        for pose in poses.dropFirst() {
            result.append(multiplied(first, inverted(pose.worldToCamera)).map(Float.init))
        }
        return result
    }

    static func multiplied(_ a: [Double], _ b: [Double]) -> [Double] {
        (0 ..< 16).map { index in
            let row = index / 4, column = index % 4
            return (0 ..< 4).reduce(0) { $0 + a[row * 4 + $1] * b[$1 * 4 + column] }
        }
    }

    /// The inverse of a 4 × 4 matrix by Gauss–Jordan elimination with partial pivoting.
    static func inverted(_ m: [Double]) -> [Double] {
        var a = m, inverse = [Double](repeating: 0, count: 16)
        for i in 0 ..< 4 { inverse[i * 4 + i] = 1 }
        for column in 0 ..< 4 {
            let pivot = (column ..< 4).max { abs(a[$0 * 4 + column]) < abs(a[$1 * 4 + column]) }!
            if pivot != column {
                for k in 0 ..< 4 {
                    a.swapAt(pivot * 4 + k, column * 4 + k)
                    inverse.swapAt(pivot * 4 + k, column * 4 + k)
                }
            }
            let scale = a[column * 4 + column]
            for k in 0 ..< 4 {
                a[column * 4 + k] /= scale
                inverse[column * 4 + k] /= scale
            }
            for row in 0 ..< 4 where row != column {
                let factor = a[row * 4 + column]
                for k in 0 ..< 4 {
                    a[row * 4 + k] -= factor * a[column * 4 + k]
                    inverse[row * 4 + k] -= factor * inverse[column * 4 + k]
                }
            }
        }
        return inverse
    }
}
