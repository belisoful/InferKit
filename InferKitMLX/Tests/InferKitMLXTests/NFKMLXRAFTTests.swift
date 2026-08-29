//
//  NFKMLXRAFTTests.swift
//  InferKitMLXTests
//
//  Feature/context encoders, an all-pairs correlation pyramid + lookup, and a ConvGRU update loop.
//  These evaluate MLX arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXRAFTTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    func testFlowFieldHasTwoChannelsAtInputSize() throws {
        try requireMLXRuntime()
        let net = NFKMLXRAFT.makeNet(iterations: 2)
        let flow = net.flow(Self.gradient(64, 64, shift: 0), Self.gradient(64, 64, shift: 2))
        eval(flow)
        XCTAssertEqual(flow.shape, [64, 64, 2], "a dense flow field at the input size")
    }

    func testACheckpointRoundTripReproducesTheFlow() throws {
        try requireMLXRuntime()
        let trained = NFKMLXRAFT.makeNet(iterations: 2)
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("raft-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = NFKMLXRAFT.makeNet(iterations: 2)
        try NFKMLXRAFT.loadWeights(into: loaded, from: url)

        let (a, b) = (Self.gradient(64, 64, shift: 0), Self.gradient(64, 64, shift: 3))
        let expected = trained.flow(a, b)
        let actual = loaded.flow(a, b)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self))
    }

    func testTheRegisteredBackendReturnsAPackedFlowMap() throws {
        try requireMLXRuntime()
        NFKMLXRAFT.register()
        let backend = try NFKMLXModelRegistry.backend(named: "raft", weightsURL: nil)
        let request = NFKInferenceRequest(inputs: ["frame0": Self.solid(64, 64, 60, 60, 60),
                                                   "frame1": Self.solid(64, 64, 200, 200, 200)])
        let output = try Self.cgImage(try backend.runInference(for: request).output(forKey: NFKOutputImage))
        XCTAssertEqual(output.width, 64)
        XCTAssertEqual(output.height, 64)
    }

    static func gradient(_ height: Int, _ width: Int, shift: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let base = (y * width + x) * 3
                values[base] = Float((x + shift) % width) / Float(width)
                values[base + 1] = Float(y) / Float(height)
                values[base + 2] = 0.5
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [1, height, width, 3]) }
    }

    static func solid(_ width: Int, _ height: Int, _ red: UInt8, _ green: UInt8, _ blue: UInt8) -> CGImage {
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        for pixel in 0 ..< (width * height) {
            pixels[pixel * 4] = red; pixels[pixel * 4 + 1] = green
            pixels[pixel * 4 + 2] = blue; pixels[pixel * 4 + 3] = 255
        }
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    static func cgImage(_ value: Any?) throws -> CGImage {
        guard let value, CFGetTypeID(value as CFTypeRef) == CGImage.typeID else { throw NFKMLXError.noOutput }
        return (value as! CGImage)
    }

    // MARK: The fused lookup

    /// The fused kernel and the elementwise gathers must produce the same numbers. This is what
    /// carries the kernel: the reference-parity record measures the whole network, and a lookup that
    /// was subtly wrong at the edges could still land inside its tolerance.
    func testTheFusedLookupMatchesTheGatherPath() throws {
        try requireMLXRuntime()
        let (h, w, c) = (8, 10, 16)
        let first = MLXRandom.normal([1, h, w, c], key: MLXRandom.key(41))
        let second = MLXRandom.normal([1, h, w, c], key: MLXRandom.key(42))
        let levels = NFKRAFTCorrelation.pyramid(first, second)

        // Coordinates deliberately off the map in places: the neighborhood radius is 4 and the
        // coarsest level is a few cells wide, so most of it lies outside and the zero padding is what
        // is actually being compared.
        let grid = MLXRandom.uniform(low: -3.0, high: Float(w) + 3.0, [1, h, w, 2],
                                     key: MLXRandom.key(43))

        let fused = NFKRAFTCorrelation.fusedLookup(levels, coords: grid, height: h, width: w)
        let gathered = NFKRAFTCorrelation.gatherLookup(levels, coords: grid, height: h, width: w)
        eval(fused, gathered)

        XCTAssertEqual(fused.shape, gathered.shape)
        let worst = abs(fused - gathered).max()
        eval(worst)
        XCTAssertLessThan(worst.item(Float.self), 1e-5,
                          "the fused kernel reproduces the gathers exactly, padding included")
    }

    /// Coordinates entirely outside the map must read as zero on both paths — the padding rule, at
    /// the extreme where it is the whole answer.
    func testAWhollyOutsideNeighborhoodReadsZeroOnBothPaths() throws {
        try requireMLXRuntime()
        let (h, w, c) = (4, 4, 8)
        let levels = NFKRAFTCorrelation.pyramid(MLXRandom.normal([1, h, w, c], key: MLXRandom.key(7)),
                                                MLXRandom.normal([1, h, w, c], key: MLXRandom.key(8)))
        let faraway = MLXArray([Float](repeating: -500, count: h * w * 2), [1, h, w, 2])

        let fused = NFKRAFTCorrelation.fusedLookup(levels, coords: faraway, height: h, width: w)
        let gathered = NFKRAFTCorrelation.gatherLookup(levels, coords: faraway, height: h, width: w)
        eval(fused, gathered)
        XCTAssertEqual(abs(fused).max().item(Float.self), 0, "nothing is sampled")
        XCTAssertEqual(abs(gathered).max().item(Float.self), 0)
    }

    /// The plane order is what the trained 1×1 `convc1` reads, and swapping the two loops would leave
    /// every value present but permuted — invisible to a magnitude check.
    func testThePlaneOrderPutsTheOuterIndexOnX() throws {
        try requireMLXRuntime()
        let (h, w, c) = (5, 5, 8)
        let levels = NFKRAFTCorrelation.pyramid(MLXRandom.normal([1, h, w, c], key: MLXRandom.key(9)),
                                                MLXRandom.normal([1, h, w, c], key: MLXRandom.key(10)))
        let grid = MLXRandom.uniform(low: 0.0, high: Float(w), [1, h, w, 2], key: MLXRandom.key(11))
        let fused = NFKRAFTCorrelation.fusedLookup(levels, coords: grid, height: h, width: w)
        let gathered = NFKRAFTCorrelation.gatherLookup(levels, coords: grid, height: h, width: w)
        eval(fused, gathered)

        // Plane by plane, not just in aggregate: a permutation preserves the total and the maximum.
        let side = 2 * kCorrRadius + 1
        for plane in [0, 1, side, side + 1, side * side - 1] {
            let a = fused[0, 0..., 0..., plane]
            let b = gathered[0, 0..., 0..., plane]
            eval(a, b)
            XCTAssertLessThan(abs(a - b).max().item(Float.self), 1e-5, "plane \(plane)")
        }
    }

    /// `lookup` must route to the fused kernel on the GPU. Measured, the fused path is about 730×
    /// faster at RAFT's own resolution (2753 ms against 3.98 ms at 60×80), so a silent fallback to
    /// the gathers would not fail anything — it would just make the model unusable.
    ///
    /// This compares bit-for-bit rather than timing, so it cannot flake under load.
    func testTheDispatchChoosesTheFusedPathOnTheGPU() throws {
        try requireMLXRuntime()
        try XCTSkipUnless(NFKMLXDevice.currentType == .gpu, "the fused path is the GPU one")
        let (h, w, c) = (6, 6, 8)
        let levels = NFKRAFTCorrelation.pyramid(MLXRandom.normal([1, h, w, c], key: MLXRandom.key(21)),
                                                MLXRandom.normal([1, h, w, c], key: MLXRandom.key(22)))
        let grid = MLXRandom.uniform(low: 0.0, high: Float(w), [1, h, w, 2], key: MLXRandom.key(23))

        let dispatched = NFKRAFTCorrelation.lookup(levels, coords: grid, height: h, width: w)
        let fused = NFKRAFTCorrelation.fusedLookup(levels, coords: grid, height: h, width: w)
        eval(dispatched, fused)
        XCTAssertEqual(abs(dispatched - fused).max().item(Float.self), 0,
                       "identical to the bit, so the kernel is what ran")
    }

    /// And the CPU stream, where a Metal kernel cannot dispatch at all, must still work.
    func testTheCPUStreamFallsBackToTheGathers() throws {
        try requireMLXRuntime()
        let (h, w, c) = (5, 5, 8)
        let levels = NFKRAFTCorrelation.pyramid(MLXRandom.normal([1, h, w, c], key: MLXRandom.key(31)),
                                                MLXRandom.normal([1, h, w, c], key: MLXRandom.key(32)))
        let grid = MLXRandom.uniform(low: 0.0, high: Float(w), [1, h, w, 2], key: MLXRandom.key(33))

        var onCPU: MLXArray?
        NFKMLXDevice.perform(on: .cpu) {
            let result = NFKRAFTCorrelation.lookup(levels, coords: grid, height: h, width: w)
            eval(result)
            onCPU = result
        }
        let reference = NFKRAFTCorrelation.gatherLookup(levels, coords: grid, height: h, width: w)
        eval(reference)
        let cpu = try XCTUnwrap(onCPU)
        XCTAssertLessThan(abs(cpu - reference).max().item(Float.self), 1e-5)
    }
}
