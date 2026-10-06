//
//  NFKMLXMatteOperationsTests.swift
//  InferKitMLXTests
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXMatteOperationsTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private var state: UInt64 = 0x2545_F491_4F6C_DD1D

    private func random() -> Float {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Float(state >> 40) / Float(1 << 24)
    }

    private func values(_ x: MLXArray) -> [Float] {
        x.asType(.float32).reshaped([-1]).asArray(Float.self)
    }

    // MARK: Levels

    func testTheClipRemapsBetweenItsLevelsAndSparesProtectedPixels() throws {
        try requireMLXRuntime()
        let alpha = MLXArray([Float(0.05), 0.1, 0.5, 0.9, 0.95]).reshaped([1, 5, 1])
        let clipped = values(NFKMLXMatteOperations.clipped(alpha, black: 0.1, white: 0.9))
        XCTAssertEqual(clipped, [0, 0, 0.5, 1, 1], accuracy: 1e-6)
        let protected = MLXArray([true, false, false, false, true]).reshaped([1, 5, 1])
        let spared = values(NFKMLXMatteOperations.clipped(alpha, black: 0.1, white: 0.9, protected: protected))
        XCTAssertEqual(spared, [0.05, 0, 0.5, 1, 0.95], accuracy: 1e-6)
    }

    func testGammaAboveOneLiftsThePartialValues() throws {
        try requireMLXRuntime()
        let lifted = values(NFKMLXMatteOperations.gammaAdjusted(MLXArray([Float(0), 0.25, 1]).reshaped([1, 3, 1]), gamma: 2))
        XCTAssertEqual(lifted[0], 0)
        XCTAssertEqual(lifted[1], 0.5, accuracy: 1e-6)
        XCTAssertEqual(lifted[2], 1)
    }

    func testEdgesMarkOnlyTheNeighborhoodOfAStep() throws {
        try requireMLXRuntime()
        let image = MLXArray((0 ..< 20).map { Float($0 < 10 ? 0.1 : 0.8) }).reshaped([1, 20, 1])
        let edges = NFKMLXMatteOperations.edges(of: image, radius: 2, tolerance: 0.3).reshaped([-1]).asArray(Bool.self)
        XCTAssertEqual(edges, (0 ..< 20).map { (8 ... 11).contains($0) })
    }

    func testTheGarbageMatteRemovesAndTheCoreMatteRestores() throws {
        try requireMLXRuntime()
        let alpha = MLXArray([Float(0.6), 0.6, 0.6]).reshaped([1, 3, 1])
        let garbage = MLXArray([Float(1), 0, 0]).reshaped([1, 3, 1])
        let core = MLXArray([Float(0), 0, 1]).reshaped([1, 3, 1])
        XCTAssertEqual(values(NFKMLXMatteOperations.masked(alpha, garbage: garbage, core: core)), [0, 0.6, 1])
    }

    // MARK: Shape

    func testTheWindowExtremeMatchesADirectSearch() throws {
        try requireMLXRuntime()
        let (height, width) = (9, 23)
        let data = (0 ..< height * width).map { _ in random() }
        let x = MLXArray(data, [height, width, 1])
        for radius in [1, 2, 3, 5, 8, 30] {
            for largest in [true, false] {
                let found = values(NFKMLXMatteOperations.windowExtreme(x, radius: radius, axis: 1, largest: largest))
                for y in 0 ..< height {
                    for column in 0 ..< width {
                        let window = data[(y * width + max(column - radius, 0)) ... (y * width + min(column + radius, width - 1))]
                        let expected = largest ? window.max()! : window.min()!
                        XCTAssertEqual(found[y * width + column], expected, "radius \(radius) at (\(y), \(column))")
                    }
                }
            }
        }
    }

    func testAGrowthIsADiscAroundEachPixelAndAShrinkItsMirror() throws {
        try requireMLXRuntime()
        let size = 15
        var data = [Float](repeating: 0, count: size * size)
        data[7 * size + 7] = 0.8
        let grown = values(NFKMLXMatteOperations.morphed(MLXArray(data, [size, size, 1]), radius: 4))
        for y in 0 ..< size {
            for x in 0 ..< size {
                let (dy, dx) = (y - 7, x - 7)
                let inside = abs(dy) <= 4 && abs(dx) <= Int(Double(16 - dy * dy).squareRoot())
                XCTAssertEqual(grown[y * size + x], inside ? 0.8 : 0, "(\(y), \(x))")
            }
        }
        let inverse = data.map { 1 - $0 }
        let shrunk = values(NFKMLXMatteOperations.morphed(MLXArray(inverse, [size, size, 1]), radius: -4))
        XCTAssertEqual(shrunk.map { 1 - $0 }, grown, accuracy: 1e-6)
        let half = values(NFKMLXMatteOperations.morphed(MLXArray(data, [size, size, 1]), radius: 0.5))
        XCTAssertEqual(half[7 * size + 8], 0.4, accuracy: 1e-6, "a half radius is half of the one-pixel growth")
    }

    func testTheFeatherKeepsAConstantMatte() throws {
        try requireMLXRuntime()
        let feathered = NFKMLXMatteOperations.feathered(MLXArray.ones([12, 17, 1]) * 0.7, radius: 6)
        XCTAssertLessThan(abs(feathered - 0.7).max().item(Float.self), 1e-6)
    }

    func testSpecksAndHolesFollowTheirAreasAndTheEdge() throws {
        try requireMLXRuntime()
        let (height, width) = (20, 30)
        var data = [Float](repeating: 0, count: height * width)
        func fill(_ rows: Range<Int>, _ columns: Range<Int>, _ value: Float) {
            for y in rows {
                for x in columns {
                    data[y * width + x] = value
                }
            }
        }
        fill(1 ..< 3, 1 ..< 3, 0.9)            // a 4-pixel speck
        fill(1 ..< 4, 10 ..< 13, 0.9)          // a 9-pixel speck
        fill(8 ..< 19, 8 ..< 28, 0.9)          // the subject ...
        fill(10 ..< 12, 10 ..< 12, 0.1)        // ... with a 4-pixel hole
        fill(13 ..< 16, 20 ..< 23, 0.1)        // ... and a 9-pixel hole
        let alpha = MLXArray(data, [height, width, 1])
        let despecked = values(NFKMLXMatteOperations.specksRemoved(alpha, minimumArea: 5))
        XCTAssertEqual(despecked[1 * width + 1], 0, "the 4-pixel speck goes")
        XCTAssertEqual(despecked[1 * width + 10], 0.9, "the 9-pixel speck stays")
        let filled = values(NFKMLXMatteOperations.holesFilled(alpha, maximumArea: 4))
        XCTAssertEqual(filled[10 * width + 10], 1, "the 4-pixel hole fills")
        XCTAssertEqual(filled[13 * width + 20], 0.1, "the 9-pixel hole stays")
        XCTAssertEqual(filled[0], 0, "the background touches the edge and stays")
    }

    func testComponentLabelsMatchAUnionFind() throws {
        try requireMLXRuntime()
        let (height, width) = (31, 47)
        let mask = (0 ..< height * width).map { _ in random() < 0.55 }
        let labels = NFKMLXMatteOperations.componentLabels(MLXArray(mask, [height, width])).reshaped([-1]).asArray(Int32.self)
        var parent = Array(0 ..< height * width)
        func root(_ i: Int) -> Int {
            var node = i
            while parent[node] != node {
                node = parent[node]
            }
            return node
        }
        for y in 0 ..< height {
            for x in 0 ..< width where mask[y * width + x] {
                for (dy, dx) in [(0, 1), (1, -1), (1, 0), (1, 1)] {
                    let (v, u) = (y + dy, x + dx)
                    if v < height, u >= 0, u < width, mask[v * width + u] {
                        let (a, b) = (root(y * width + x), root(v * width + u))
                        parent[max(a, b)] = min(a, b)
                    }
                }
            }
        }
        for index in mask.indices {
            XCTAssertEqual(labels[index], mask[index] ? Int32(root(index)) : Int32.max, "pixel \(index)")
        }
    }

    // MARK: Refinement

    func testTheGuidedFilterMatchesADirectEvaluation() throws {
        try requireMLXRuntime()
        let (height, width, radius) = (12, 14, 2)
        let epsilon: Float = 1e-3
        let guide = (0 ..< height * width * 3).map { _ in random() }
        let alpha = (0 ..< height * width).map { _ in random() }
        let filtered = values(NFKMLXMatteOperations.guidedFiltered(MLXArray(alpha, [height, width, 1]),
                                                                   guide: MLXArray(guide, [height, width, 3]),
                                                                   radius: radius, epsilon: epsilon, subsampling: 1))
        func window(_ y: Int, _ x: Int) -> [Int] {
            var indices = [Int]()
            for v in max(y - radius, 0) ... min(y + radius, height - 1) {
                for u in max(x - radius, 0) ... min(x + radius, width - 1) {
                    indices.append(v * width + u)
                }
            }
            return indices
        }
        var a = [[Double]](repeating: [0, 0, 0], count: height * width)
        var b = [Double](repeating: 0, count: height * width)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let indices = window(y, x)
                let n = Double(indices.count)
                var mean = [0.0, 0, 0], meanP = 0.0, corr = [0.0, 0, 0]
                var sigma = [[Double]](repeating: [0, 0, 0], count: 3)
                for i in indices {
                    meanP += Double(alpha[i]) / n
                    for c in 0 ..< 3 {
                        mean[c] += Double(guide[i * 3 + c]) / n
                        corr[c] += Double(guide[i * 3 + c]) * Double(alpha[i]) / n
                        for d in 0 ..< 3 {
                            sigma[c][d] += Double(guide[i * 3 + c]) * Double(guide[i * 3 + d]) / n
                        }
                    }
                }
                for c in 0 ..< 3 {
                    for d in 0 ..< 3 {
                        sigma[c][d] -= mean[c] * mean[d]
                    }
                    sigma[c][c] += Double(epsilon)
                }
                let cov = (0 ..< 3).map { corr[$0] - mean[$0] * meanP }
                let solved = solve3(sigma, cov)
                a[y * width + x] = solved
                b[y * width + x] = meanP - (0 ..< 3).reduce(0) { $0 + solved[$1] * mean[$1] }
            }
        }
        for y in 0 ..< height {
            for x in 0 ..< width {
                let indices = window(y, x)
                let n = Double(indices.count)
                var q = indices.reduce(0) { $0 + b[$1] } / n
                for c in 0 ..< 3 {
                    q += indices.reduce(0) { $0 + a[$1][c] } / n * Double(guide[(y * width + x) * 3 + c])
                }
                XCTAssertEqual(Double(filtered[y * width + x]), min(max(q, 0), 1), accuracy: 2e-4, "(\(y), \(x))")
            }
        }
    }

    private func solve3(_ m: [[Double]], _ v: [Double]) -> [Double] {
        func det(_ m: [[Double]]) -> Double {
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        }
        let d = det(m)
        return (0 ..< 3).map { column in
            var replaced = m
            for row in 0 ..< 3 {
                replaced[row][column] = v[row]
            }
            return det(replaced) / d
        }
    }

    func testTheEstimatedColorsRecompositeTheImageAndRecoverTheForeground() throws {
        try requireMLXRuntime()
        let (height, width) = (40, 64)
        var image = [Float](), foreground = [Float](), alpha = [Float]()
        for y in 0 ..< height {
            for x in 0 ..< width {
                let a = min(max(Float(x - 20) / 24, 0), 1)
                let f: [Float] = [0.8, 0.3 + 0.2 * Float(y) / Float(height), 0.2]
                let b: [Float] = [0.1, 0.7, 0.15]
                alpha.append(a)
                foreground += f
                image += (0 ..< 3).map { a * f[$0] + (1 - a) * b[$0] }
            }
        }
        let estimate = NFKMLXMatteOperations.estimatedColors(image: MLXArray(image, [height, width, 3]),
                                                             alpha: MLXArray(alpha, [height, width, 1]))
        let a = MLXArray(alpha, [height, width, 1])
        let recomposed = estimate.foreground * a + estimate.background * (1 - a)
        XCTAssertLessThan(abs(recomposed - MLXArray(image, [height, width, 3])).mean().item(Float.self), 0.01)
        let partial = ((a .> Float(0.2)) .&& (a .< Float(0.8))).asType(.float32)
        let truth = MLXArray(foreground, [height, width, 3])
        let estimateError = (abs(estimate.foreground - truth) * partial).sum().item(Float.self)
        let imageError = (abs(MLXArray(image, [height, width, 3]) - truth) * partial).sum().item(Float.self)
        XCTAssertLessThan(estimateError, imageError * 0.25, "the estimate is far closer to the foreground than the plate")
    }

    // MARK: Compositing and time

    func testTheLightWrapTouchesOnlyTheEdge() throws {
        try requireMLXRuntime()
        let width = 40
        let alpha = MLXArray((0 ..< width).map { Float($0 < 20 ? 1 : 0) }).reshaped([1, width, 1])
        let foreground = MLXArray.ones([1, width, 3]) * MLXArray([Float(0.2), 0.2, 0.2])
        let background = MLXArray.ones([1, width, 3]) * MLXArray([Float(0.9), 0.5, 0.1])
        let wrapped = NFKMLXMatteOperations.lightWrapped(foreground: foreground, alpha: alpha, background: background,
                                                         radius: 4, strength: 1)
        XCTAssertLessThan(abs(wrapped[0, 0] - foreground[0, 0]).max().item(Float.self), 1e-6, "far inside is untouched")
        XCTAssertLessThan(abs(wrapped[0, 39] - background[0, 39]).max().item(Float.self), 1e-6, "outside is the background")
        XCTAssertGreaterThan(wrapped[0, 19, 0].item(Float.self), 0.25, "the edge takes the background's light")
        let none = NFKMLXMatteOperations.lightWrapped(foreground: foreground, alpha: alpha, background: background,
                                                      radius: 4, strength: 0)
        XCTAssertLessThan(abs(none - (foreground * alpha + background * (1 - alpha))).max().item(Float.self), 1e-6)
    }

    func testTheAdditiveKeyAddsThePlatesDepartureFromTheScreen() throws {
        try requireMLXRuntime()
        let screen = MLXArray([Float(0.1), 0.7, 0.15])
        let plate = MLXArray([Float(0.1), 0.7, 0.15, 0.3, 0.8, 0.35]).reshaped([1, 2, 3])
        let alpha = MLXArray.zeros([1, 2, 1])
        let background = MLXArray.ones([1, 2, 3]) * 0.4
        let keyed = values(NFKMLXMatteOperations.additiveKeyed(foreground: MLXArray.zeros([1, 2, 3]), alpha: alpha,
                                                               background: background, plate: plate,
                                                               screenColor: screen, saturation: 1, gain: 1))
        XCTAssertEqual(Array(keyed[0 ..< 3]), [0.4, 0.4, 0.4], accuracy: 1e-6, "the screen itself adds nothing")
        XCTAssertEqual(Array(keyed[3 ..< 6]), [0.6, 0.5, 0.6], accuracy: 1e-6, "a lighter strand adds its difference")
    }

    func testTheSourcePassesThroughOnlyWhereTheMatteIsSolid() throws {
        try requireMLXRuntime()
        let foreground = MLXArray.zeros([1, 3, 3])
        let plate = MLXArray.ones([1, 3, 3])
        let alpha = MLXArray([Float(0.5), 0.975, 1]).reshaped([1, 3, 1])
        let passed = values(NFKMLXMatteOperations.sourcePassedThrough(foreground: foreground, plate: plate, alpha: alpha,
                                                                      threshold: 0.95))
        XCTAssertEqual(passed[0], 0)
        XCTAssertEqual(passed[3], 0.5, accuracy: 1e-5)
        XCTAssertEqual(passed[6], 1)
    }

    func testTheTemporalBlendStopsWhereTheImageMoves() throws {
        try requireMLXRuntime()
        let alpha = MLXArray([Float(1), 1]).reshaped([1, 2, 1])
        let previous = MLXArray([Float(0), 0]).reshaped([1, 2, 1])
        let image = MLXArray([Float(0.5), 0.5, 0.5, 0.5, 0.5, 0.5]).reshaped([1, 2, 3])
        let previousImage = MLXArray([Float(0.52), 0.5, 0.5, 0.9, 0.5, 0.5]).reshaped([1, 2, 3])
        let blended = values(NFKMLXMatteOperations.temporallyBlended(alpha, previous: previous, image: image,
                                                                     previousImage: previousImage, weight: 0.25))
        XCTAssertEqual(blended[0], 0.75, accuracy: 1e-6, "a still pixel moves a quarter toward the last frame")
        XCTAssertEqual(blended[1], 1, "a moving pixel keeps its own matte")
    }
}

private func XCTAssertEqual(_ a: [Float], _ b: [Float], accuracy: Float, _ message: String = "",
                            file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(a.count, b.count, message, file: file, line: line)
    for (x, y) in zip(a, b) {
        XCTAssertEqual(x, y, accuracy: accuracy, message, file: file, line: line)
    }
}

final class NFKMLXMatteRefinerTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    func testANewRefinerReturnsTheMatteItIsGiven() throws {
        try requireMLXRuntime()
        let alpha = MLXArray((0 ..< 30).map { Float($0) / 29 }).reshaped([5, 6, 1])
        let plate = MLXArray.ones([5, 6, 3]) * 0.5
        XCTAssertEqual(abs(NFKMLXMatteRefiner().refined(alpha, plate: plate) - alpha).max().item(Float.self), 0)
    }

    func testTheRefinerClipsBeforeItRemovesSpecks() throws {
        try requireMLXRuntime()
        var data = [Float](repeating: 0, count: 10 * 10)
        data[0] = 0.4
        data[1] = 0.4
        data[55] = 0.9
        let refiner = NFKMLXMatteRefiner()
        refiner.clipBlack = 0.1
        refiner.clipWhite = 0.5
        refiner.minimumSpeckArea = 2
        let refined = refiner.refined(MLXArray(data, [10, 10, 1]), plate: MLXArray.zeros([10, 10, 3]))
            .reshaped([-1]).asArray(Float.self)
        // The clip lifts the 0.4 pair to 0.75, above half, so the pair survives as a 2-pixel island;
        // the single 0.9 pixel is a 1-pixel island and goes.
        XCTAssertEqual(refined[0], 0.75, accuracy: 1e-6)
        XCTAssertEqual(refined[55], 0)
    }

    func testTheObjectiveCEntryRefinesImages() throws {
        try requireMLXRuntime()
        let matte = try NFKMLXImageBridge.cgImage(from: MLXArray.ones([8, 12, 1]) * 0.6, options: NFKMLXImageOptions())
        let plate = try NFKMLXImageBridge.cgImage(from: MLXArray.ones([8, 12, 3]) * 0.3, options: NFKMLXImageOptions())
        let garbage = try NFKMLXImageBridge.cgImage(from: MLXArray.ones([8, 12, 1]), options: NFKMLXImageOptions())
        let refined = try NFKMLXMatteRefiner().refine(matte: matte, plate: plate, garbageMatte: garbage, coreMatte: nil)
        XCTAssertEqual(refined.width, 12)
        let values = try NFKMLXImageBridge.tensor(from: refined, channels: 1, colorSpace: CGColorSpaceCreateDeviceRGB())
        XCTAssertEqual(values.max().item(Float.self), 0, "the garbage matte covers the whole frame")
        let decontaminated = try NFKMLXMatteRefiner().decontaminatedForeground(plate: plate, matte: matte)
        XCTAssertEqual(decontaminated.height, 8)
        let composite = try NFKMLXMatteRefiner().composite(foreground: plate, matte: matte, background: plate,
                                                           lightWrapRadius: 3, lightWrapStrength: 0.5)
        XCTAssertEqual(composite.width, 12)
    }

    func testTheTemporalBlenderPassesTheFirstFrameAndForgetsAtAReset() throws {
        try requireMLXRuntime()
        let blender = NFKMLXMatteTemporalBlender()
        blender.weight = 0.5
        let plate = MLXArray.ones([2, 2, 3]) * 0.5
        let first = blender.blended(MLXArray.ones([2, 2, 1]), plate: plate)
        XCTAssertEqual(first.min().item(Float.self), 1)
        let second = blender.blended(MLXArray.zeros([2, 2, 1]), plate: plate)
        XCTAssertEqual(second.max().item(Float.self), 0.5, accuracy: 1e-6)
        blender.reset()
        let afterCut = blender.blended(MLXArray.zeros([2, 2, 1]), plate: plate)
        XCTAssertEqual(afterCut.max().item(Float.self), 0)
    }
}
