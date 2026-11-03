//
//  NFKMLXLTX2DFRLayout.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX

// The canvas layout of LTX-2.5's Diffusion Fidelity Rendering (DFR): the keyframe segment grid, the
// temporal refine round's seam tiles, the spatial epilogue's tiles, and the token plan a tiled transformer
// pass walks; and the explicit token coordinates the DFR passes hand the transformer, since their packed
// sequence appends keyframe, slot, and reference tokens after the latent grid.
//
// Frame positions are pixel frames unless a name says latent. A keyframe sits on a latent border, so its
// pixel position is a multiple of the autoencoder's temporal compression.

/// The keyframe segment grid and the tilings DFR cuts it into, the reference's `dfr_layout`.
public enum NFKMLXLTX2DFRLayout {

    /// The candidate segment lengths, in latent frames: 24 and 32 pixel frames at a compression of 8.
    static let segmentLatentCandidates = [3, 4]

    /// The overlap, in latent cells, between the spatial epilogue's height and width tiles.
    static let epilogueSpatialOverlap = 12

    /// The segment length, in pixel frames, that pads `contentFrames` (`num_frames − 1`) least; a tie keeps
    /// the longer segment.
    public static func segmentLength(contentFrames: Int, temporalCompression r: Int = 8) -> Int {
        segmentLatentCandidates.map { $0 * r }.max { a, b in
            let padA = (a - contentFrames % a) % a, padB = (b - contentFrames % b) % b
            return (-padA, a) < (-padB, b)
        }!
    }

    /// `num_frames − 1` padded to whole segments: the padded frame count, the segment length, and the
    /// keyframe positions `[S, 2 S, …, N′ − 1]` (frame 0 excluded, the last frame included).
    public static func canvas(frames: Int, temporalCompression r: Int = 8) throws
        -> (frames: Int, segment: Int, positions: [Int]) {
        guard (frames - 1) % r == 0 else {
            throw NFKMLXError.unsupportedConfiguration("a DFR canvas is one more than a multiple of \(r) frames, not \(frames)")
        }
        let content = frames - 1
        guard content >= r else {
            throw NFKMLXError.unsupportedConfiguration("a DFR canvas holds at least \(r + 1) frames")
        }
        let segment = segmentLength(contentFrames: content, temporalCompression: r)
        let padded = content + (segment - content % segment) % segment
        return (padded + 1, segment, (1 ... padded / segment).map { $0 * segment })
    }

    /// One tile's half-open extent on an axis and the run-ups it shares with its neighbours.
    public struct Interval: Equatable, Sendable {
        public var start: Int
        public var end: Int
        public var leftRamp: Int
        public var rightRamp: Int
    }

    /// Splits an axis on segment edges `seams` (`0 … dimSize − 1`) into at most `tiles` tiles, the leftover
    /// segments to the leading tiles. Every tile but the first reaches `overlap` cells back before the edge it
    /// resumes after; that run-up is its left ramp.
    public static func splitAtSeams(_ seams: [Int], tiles requested: Int, overlap: Int, dimSize: Int) throws -> [Interval] {
        guard overlap >= 0, seams.count >= 2, seams[0] == 0, seams.last == dimSize - 1,
              zip(seams, seams.dropFirst()).allSatisfy({ $0 < $1 }), requested >= 1 else {
            throw NFKMLXError.unsupportedConfiguration("DFR seams must rise from 0 to the last cell, \(seams) over \(dimSize)")
        }
        let segments = seams.count - 1
        let tiles = min(requested, segments)
        let (base, remainder) = segments.quotientAndRemainder(dividingBy: tiles)
        var intervals = [Interval](), cursor = 0
        for index in 0 ..< tiles {
            let resume = seams[cursor] + 1
            let start = index == 0 ? 0 : max(0, resume - overlap)
            cursor += base + (index < remainder ? 1 : 0)
            intervals.append(Interval(start: start, end: seams[cursor] + 1, leftRamp: index == 0 ? 0 : resume - start,
                                      rightRamp: 0))
        }
        return intervals
    }

    /// One temporal refine tile: its latent interval, its inclusive pixel bounds, the carried seam keyframes
    /// inside it, and the mid-segment slots it invents.
    public struct TemporalTile: Equatable, Sendable {
        public var interval: Interval
        public var pixelStart: Int
        public var pixelEnd: Int
        public var anchors: [Int]
        public var slots: [Int]
    }

    /// The seam tiles of one temporal round over a canvas of `frames` pixel frames whose keyframes sit at
    /// `seamPositions`. The run-up is one segment plus the shared seam cell.
    public static func temporalTiles(seamPositions: [Int], frames: Int, tiles: Int, temporalCompression r: Int = 8) throws
        -> [TemporalTile] {
        let seams = [0] + seamPositions.map { $0 / r }
        guard seamPositions.allSatisfy({ $0 % r == 0 }) else {
            throw NFKMLXError.unsupportedConfiguration("DFR keyframes sit on latent borders, multiples of \(r)")
        }
        let latentLength = (frames - 1) / r + 1
        let overlap = seams[1] - seams[0] + 1
        return try splitAtSeams(seams, tiles: tiles, overlap: overlap, dimSize: latentLength).map { interval in
            let pixelStart = interval.start * r, pixelEnd = (interval.end - 1) * r
            let anchors = seamPositions.filter { pixelStart <= $0 && $0 <= pixelEnd }
            let marks = [pixelStart] + seamPositions.filter { pixelStart < $0 && $0 <= pixelEnd }
            let slots = zip(marks, marks.dropFirst()).map { ($0 + $1) / 2 }
            return TemporalTile(interval: interval, pixelStart: pixelStart, pixelEnd: pixelEnd, anchors: anchors, slots: slots)
        }
    }

    /// `tiles` evenly sized tiles over `dimSize` cells sharing `overlap` cells with each neighbour, the leading
    /// tiles taking the remainder.
    public static func splitByCount(dimSize: Int, tiles: Int, overlap: Int) throws -> [Interval] {
        guard tiles > 1 else { return [Interval(start: 0, end: dimSize, leftRamp: 0, rightRamp: 0)] }
        let total = dimSize + overlap * (tiles - 1)
        let (size, remainder) = total.quotientAndRemainder(dividingBy: tiles)
        guard size > overlap else {
            throw NFKMLXError.unsupportedConfiguration("a tile of \(size) cells cannot hold an overlap of \(overlap)")
        }
        let stride = size - overlap
        return (0 ..< tiles).map { index in
            let shift = min(index, remainder), grow = index < remainder ? 1 : 0
            return Interval(start: index * stride + shift, end: index * stride + size + shift + grow,
                            leftRamp: index == 0 ? 0 : overlap, rightRamp: index == tiles - 1 ? 0 : overlap)
        }
    }

    /// A blend weight that fades in over `leftRamp` cells and out over `rightRamp`, the interior of a linspace,
    /// so two tiles sharing `k` cells sum to one there.
    public static func trapezoid(length: Int, leftRamp: Int, rightRamp: Int) -> [Float] {
        var mask = [Float](repeating: 1, count: length)
        if leftRamp > 0 {
            let ramp = linspace(0, 1, leftRamp + 2)
            for i in 0 ..< leftRamp { mask[i] *= ramp[i + 1] }
        }
        if rightRamp > 0 {
            let ramp = linspace(1, 0, rightRamp + 2)
            for i in 0 ..< rightRamp { mask[length - rightRamp + i] *= ramp[i + 1] }
        }
        return mask
    }

    /// `torch.linspace(start, end, n)` in float32: the first half stepped from `start`, the rest back from `end`.
    static func linspace(_ start: Float, _ end: Float, _ n: Int) -> [Float] {
        let step = (end - start) / Float(n - 1)
        return (0 ..< n).map { i in i < n / 2 ? start + step * Float(i) : end - step * Float(n - 1 - i) }
    }

    /// A weight that drops the first `leftRamp` cells and keeps the rest whole: a seam cut.
    public static func rectangle(length: Int, leftRamp: Int) -> [Float] {
        (0 ..< length).map { $0 < leftRamp ? 0 : 1 }
    }

    /// One spatial epilogue tile: half-open latent extents and the separable `[F, H, W]` blend window.
    public struct EpilogueTile {
        public var frames: Range<Int>
        public var heights: Range<Int>
        public var widths: Range<Int>
        public var weight: MLXArray
    }

    /// The epilogue's tiling of a latent grid `(F, H, W)`: height and width split in two with a 12-cell
    /// trapezoidal overlap, frames into `frameTiles` cut on the interior `frameSeams` (or blended where none is
    /// interior). An axis too short for its tiles runs untiled and an overlap it cannot hold is clamped.
    public static func epilogueTiles(latentShape: (frames: Int, height: Int, width: Int), frameTiles: Int,
                                     frameSeams: [Int]) throws -> [EpilogueTile] {
        let shape = [latentShape.frames, latentShape.height, latentShape.width]
        let frameOverlap = frameTiles > 1 && !frameSeams.isEmpty ? frameSeams[0] + 1 : 0
        var configs = [(Int, Int, Int)]()
        for (axis, (tiles, overlap)) in [(frameTiles, frameOverlap), (2, epilogueSpatialOverlap),
                                         (2, epilogueSpatialOverlap)].enumerated() {
            let size = shape[axis]
            if tiles > 1, size < tiles {
                configs.append((size, 1, 0))
            } else if tiles > 1, overlap > size - tiles {
                configs.append((size, tiles, size - tiles))
            } else {
                configs.append((size, tiles, overlap))
            }
        }
        let interior = Array(Set(frameSeams.filter { 0 < $0 && $0 < shape[0] - 1 })).sorted()
        let seamCut = configs[0].1 > 1 && !interior.isEmpty
        let frameIntervals = seamCut
            ? try splitAtSeams([0] + interior + [shape[0] - 1], tiles: configs[0].1, overlap: configs[0].2, dimSize: shape[0])
            : try splitByCount(dimSize: shape[0], tiles: configs[0].1, overlap: configs[0].2)
        let heightIntervals = try splitByCount(dimSize: configs[1].0, tiles: configs[1].1, overlap: configs[1].2)
        let widthIntervals = try splitByCount(dimSize: configs[2].0, tiles: configs[2].1, overlap: configs[2].2)
        var tiles = [EpilogueTile]()
        for f in frameIntervals {
            for h in heightIntervals {
                for w in widthIntervals {
                    let frameMask = seamCut ? rectangle(length: f.end - f.start, leftRamp: f.leftRamp)
                        : trapezoid(length: f.end - f.start, leftRamp: f.leftRamp, rightRamp: f.rightRamp)
                    let heightMask = trapezoid(length: h.end - h.start, leftRamp: h.leftRamp, rightRamp: h.rightRamp)
                    let widthMask = trapezoid(length: w.end - w.start, leftRamp: w.leftRamp, rightRamp: w.rightRamp)
                    let weight = MLXArray(frameMask).reshaped([-1, 1, 1]) * MLXArray(heightMask).reshaped([1, -1, 1])
                        * MLXArray(widthMask).reshaped([1, 1, -1])
                    tiles.append(EpilogueTile(frames: f.start ..< f.end, heights: h.start ..< h.end,
                                              widths: w.start ..< w.end, weight: weight))
                }
            }
        }
        return tiles
    }

    /// One tile's share of a packed token sequence: the token indices it processes (grid tokens first), one
    /// blend weight per token, and those tokens' coordinates rebased on the tile's first grid token.
    public struct TokenPlan {
        public var keep: [Int]
        public var weights: MLXArray
        public var coordinates: NFKMLXLTX2VideoCoordinates
    }

    /// Resolves `tiles` against a pass's coordinates: each tile takes its window's grid tokens and every
    /// appended token whose interval overlaps that window on all three axes; an appended token kept by `n`
    /// tiles weighs `1 / n`.
    public static func tokenPlan(_ tiles: [EpilogueTile], coordinates: NFKMLXLTX2VideoCoordinates,
                                 latentFrames: Int, latentHeight: Int, latentWidth: Int) throws -> [TokenPlan] {
        let perFrame = latentHeight * latentWidth
        let gridTokens = latentFrames * perFrame
        let appended = gridTokens ..< coordinates.count
        var bases = [[Int]](), starts = [[Float]](), kept = [[Bool]]()
        for tile in tiles {
            var base = [Int]()
            for f in tile.frames { for h in tile.heights { for w in tile.widths { base.append(f * perFrame + h * latentWidth + w) } } }
            bases.append(base)
            let tileStarts = (0 ..< 3).map { axis in base.map { coordinates.starts[axis][$0] }.min()! }
            let tileEnds = (0 ..< 3).map { axis in base.map { coordinates.ends[axis][$0] }.max()! }
            starts.append(tileStarts)
            kept.append(appended.map { token in
                (0 ..< 3).allSatisfy { coordinates.starts[$0][token] < tileEnds[$0] && coordinates.ends[$0][token] > tileStarts[$0] }
            })
        }
        let keepers = appended.indices.map { offset in kept.reduce(0) { $0 + ($1[offset - gridTokens] ? 1 : 0) } }
        if let unclaimed = keepers.firstIndex(of: 0) {
            throw NFKMLXError.unsupportedConfiguration(
                "conditioning token \(gridTokens + unclaimed) lies outside every epilogue tile")
        }
        return tiles.indices.map { index in
            let conditions = appended.filter { kept[index][$0 - gridTokens] }
            let keep = bases[index] + conditions
            let weights = concatenated([tiles[index].weight.reshaped([-1]),
                                        MLXArray(conditions.map { 1 / Float(keepers[$0 - gridTokens]) })])
            return TokenPlan(keep: keep, weights: weights,
                             coordinates: coordinates.subset(keep).shifted(by: starts[index]))
        }
    }

    /// The latent frames covering `frames` pixel frames: a canvas with its padded tail dropped.
    public static func trimmedLatentFrames(frames: Int, temporalCompression r: Int = 8) -> Int { (frames - 1) / r + 1 }
}

/// Each video token's half-open interval on (time in seconds, height in pixels, width in pixels), the
/// reference's `video_coords`, for a pass whose token sequence is not one latent grid.
public struct NFKMLXLTX2VideoCoordinates: Equatable, Sendable {
    /// `[axis][token]` interval starts and ends.
    public var starts: [[Float]]
    public var ends: [[Float]]

    public init(starts: [[Float]], ends: [[Float]]) {
        self.starts = starts
        self.ends = ends
    }

    public var count: Int { starts[0].count }

    /// The latent grid's intervals, the reference's `prepare_video_coords`: pixel extents, the frame axis
    /// shifted back by the causal first frame's stride, clamped at zero, and in seconds.
    public static func grid(frames: Int, height: Int, width: Int, configuration c: NFKMLXLTX2Configuration,
                            fps: Float) -> NFKMLXLTX2VideoCoordinates {
        let scales = c.vaeScaleFactors.map(Float.init)
        let patches = [c.patchSizeT, c.patchSize, c.patchSize]
        var starts = [[Float]](repeating: [], count: 3), ends = [[Float]](repeating: [], count: 3)
        for f in stride(from: 0, to: frames, by: c.patchSizeT) {
            for h in stride(from: 0, to: height, by: c.patchSize) {
                for w in stride(from: 0, to: width, by: c.patchSize) {
                    let cell = [Float(f), Float(h), Float(w)]
                    for axis in 0 ..< 3 {
                        var begin = cell[axis] * scales[axis], end = (cell[axis] + Float(patches[axis])) * scales[axis]
                        if axis == 0 {
                            begin = max(0, begin + Float(c.causalOffset) - scales[0]) / fps
                            end = max(0, end + Float(c.causalOffset) - scales[0]) / fps
                        }
                        starts[axis].append(begin)
                        ends[axis].append(end)
                    }
                }
            }
        }
        return NFKMLXLTX2VideoCoordinates(starts: starts, ends: ends)
    }

    /// An appended keyframe's intervals, the reference's `_prepare_keyframe_coords`: no causal shift, the frame
    /// axis offset to `pixelFrame`, a single-pixel-frame keyframe one pixel frame long, in seconds.
    public static func keyframe(frames: Int, height: Int, width: Int, pixelFrame: Int, pixelFrames: Int,
                                configuration c: NFKMLXLTX2Configuration, fps: Float) -> NFKMLXLTX2VideoCoordinates {
        let scales = c.vaeScaleFactors.map(Float.init)
        let patches = [c.patchSizeT, c.patchSize, c.patchSize]
        var starts = [[Float]](repeating: [], count: 3), ends = [[Float]](repeating: [], count: 3)
        for f in stride(from: 0, to: frames, by: c.patchSizeT) {
            for h in stride(from: 0, to: height, by: c.patchSize) {
                for w in stride(from: 0, to: width, by: c.patchSize) {
                    let cell = [Float(f), Float(h), Float(w)]
                    for axis in 0 ..< 3 {
                        var begin = cell[axis] * scales[axis], end = (cell[axis] + Float(patches[axis])) * scales[axis]
                        if axis == 0 {
                            begin += Float(pixelFrame)
                            end = pixelFrames == 1 ? begin + 1 : end + Float(pixelFrame)
                            begin /= fps
                            end /= fps
                        }
                        starts[axis].append(begin)
                        ends[axis].append(end)
                    }
                }
            }
        }
        return NFKMLXLTX2VideoCoordinates(starts: starts, ends: ends)
    }

    /// `[self, others…]` as one sequence.
    public func appending(_ others: [NFKMLXLTX2VideoCoordinates]) -> NFKMLXLTX2VideoCoordinates {
        var result = self
        for other in others {
            for axis in 0 ..< 3 {
                result.starts[axis] += other.starts[axis]
                result.ends[axis] += other.ends[axis]
            }
        }
        return result
    }

    /// The spatial axes scaled by `factor`, the IC-LoRA reference mapped into the target's coordinates.
    public func spatiallyScaled(by factor: Float) -> NFKMLXLTX2VideoCoordinates {
        var result = self
        for axis in 1 ..< 3 {
            result.starts[axis] = starts[axis].map { $0 * factor }
            result.ends[axis] = ends[axis].map { $0 * factor }
        }
        return result
    }

    func subset(_ tokens: [Int]) -> NFKMLXLTX2VideoCoordinates {
        NFKMLXLTX2VideoCoordinates(starts: starts.map { axis in tokens.map { axis[$0] } },
                                   ends: ends.map { axis in tokens.map { axis[$0] } })
    }

    func shifted(by offsets: [Float]) -> NFKMLXLTX2VideoCoordinates {
        NFKMLXLTX2VideoCoordinates(starts: (0 ..< 3).map { axis in starts[axis].map { $0 - offsets[axis] } },
                                   ends: (0 ..< 3).map { axis in ends[axis].map { $0 - offsets[axis] } })
    }

    /// The rotary positions: each interval's midpoint over the axis's base extent.
    func positions(_ c: NFKMLXLTX2Configuration) -> [[Float]] {
        let extents = [Float(c.posEmbedMaxPos), Float(c.baseHeight), Float(c.baseWidth)]
        return (0 ..< 3).map { axis in
            zip(starts[axis], ends[axis]).map { ($0 + $1) / 2 / extents[axis] }
        }
    }
}
