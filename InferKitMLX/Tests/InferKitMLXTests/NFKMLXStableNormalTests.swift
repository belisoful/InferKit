//
//  NFKMLXStableNormalTests.swift
//  InferKitMLXTests
//
//  StableNormal-turbo. The structural tests run tiny configurations with random weights. The parity tests
//  read the release directory (IK_VAL_YOSO_NORMAL) and the Hi3DGen oracle record (IK_PARITY_HI3DGEN, from
//  `run_reference.py hi3dgen`, a float32 CPU run), and skip without them. The seam test also reads the floor
//  record (IK_PARITY_HI3DGEN_FLOOR_NORMAL, from `hi3dgen_floor_normal`: the same step in float64 on the
//  record's inputs) and holds each seam within twice the reference's own float32 spread.
//

import XCTest
import InferKit
import MLX
import MLXNN
import CoreGraphics
@testable import InferKitMLX

final class NFKMLXStableNormalTests: XCTestCase {

    /// The share of the 8-bit map's bytes the run may differ in, each by one level.
    private static let mapByteShare = 0.001
    private static let normalImageByteShare = 0.001
    /// The per-pixel renormalization alone, from the recorded decode (measured 0.9999999999999688).
    private static let renormalizationBar = 0.9999999999999

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func envPath(_ key: String) throws -> URL {
        guard let path = NFKMLXValidationConfig.environment[key], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set \(key)")
        }
        return URL(fileURLWithPath: path)
    }

    private func record() throws -> [String: MLXArray] {
        try MLX.loadArrays(url: try envPath("IK_PARITY_HI3DGEN"))
    }

    /// Writes a measured figure to standard error, unbuffered, so a later crash in the process keeps it.
    private func report(_ line: String) {
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }

    private func tensor(_ record: [String: MLXArray], _ key: String) throws -> MLXArray {
        guard let value = record[key] else {
            throw XCTSkip("the record holds no \(key)")
        }
        return value
    }

    private func bytes(_ a: MLXArray) -> [UInt8] { a.asType(.uint8).reshaped([-1]).asArray(UInt8.self) }

    private func floats(_ a: MLXArray) -> [Float] { a.asType(.float32).reshaped([-1]).asArray(Float.self) }

    private func channelsLast(_ a: MLXArray) -> MLXArray { a.transposed(0, 2, 3, 1) }

    private func cosine(_ x: [Float], _ y: [Float]) -> Double {
        XCTAssertEqual(x.count, y.count)
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< min(x.count, y.count) {
            dot += Double(x[i]) * Double(y[i]); na += Double(x[i]) * Double(x[i]); nb += Double(y[i]) * Double(y[i])
        }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-30)
    }

    private func assertCosine(_ value: MLXArray, _ reference: MLXArray, _ seam: String, bar: Double) {
        let x = floats(value), y = floats(reference)
        let figure = cosine(x, y)
        let worst = zip(x, y).map { abs($0 - $1) }.max() ?? 0
        report("StableNormal \(seam): cosine \(figure), max |difference| \(worst)")
        XCTAssertGreaterThanOrEqual(figure, bar, seam)
    }

    /// One minus the cosine, as half the squared distance between the unit vectors, which keeps the floors'
    /// last digits where the plain 1 − a·b / (|a| |b|) loses them to its sums' rounding.
    private func separation(_ x: [Double], _ y: [Double]) -> Double {
        XCTAssertEqual(x.count, y.count)
        var na = 0.0, nb = 0.0
        for i in 0 ..< min(x.count, y.count) {
            na += x[i] * x[i]
            nb += y[i] * y[i]
        }
        let sa = 1 / na.squareRoot(), sb = 1 / nb.squareRoot()
        var sum = 0.0
        for i in 0 ..< min(x.count, y.count) {
            let d = x[i] * sa - y[i] * sb
            sum += d * d
        }
        return sum / 2
    }

    private func doubles(_ a: MLXArray) -> [Double] { floats(a).map(Double.init) }

    /// Holds `value` (channels last) within twice the float32 record's own separation from the float64 result
    /// at `seam`, both read channels first.
    private func assertWithinFloor(_ value: MLXArray, _ floors: [String: MLXArray], _ parity: [String: MLXArray],
                                   _ seam: String, recorded: String, label: String) {
        guard let rounded = floors[seam], let residual = floors[seam + ".residual"], let reference = parity[recorded] else {
            return XCTFail("the floor or parity record holds no \(seam) and its residual")
        }
        let exact = zip(doubles(rounded), doubles(residual)).map { $0 + $1 }
        let floor = separation(doubles(reference), exact)
        let port = doubles(value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        let gap = separation(port, exact)
        report("StableNormal floor \(label): port \(gap), reference \(floor), ratio \(gap / floor)")
        XCTAssertLessThanOrEqual(gap, 2 * floor, label)
    }

    /// The count of differing bytes and the largest difference.
    private func byteDifferences(_ x: [UInt8], _ y: [UInt8]) -> (count: Int, largest: Int) {
        XCTAssertEqual(x.count, y.count)
        var count = 0, largest = 0
        for (a, b) in zip(x, y) where a != b {
            count += 1
            largest = max(largest, abs(Int(a) - Int(b)))
        }
        return (count, largest)
    }

    // MARK: - Structure

    /// Two levels of 32 and 64 channels, one layer each, attention at the first, over a two-level autoencoder.
    private func tinyNet() -> NFKMLXStableNormalNet {
        var unet = NFKMLXSDUNetConfiguration.yoso
        unet.blockChannels = [32, 64]
        unet.layersPerBlock = 1
        unet.attends = [true, false]
        unet.attentionHeads = [2, 2]
        unet.onlyCrossAttention = [false, false]
        unet.transformerLayers = [1, 1]
        unet.crossAttentionDimensions = 16
        unet.normalizationGroups = 8
        var vae = NFKMLXSDVAEConfiguration.stableDiffusion
        vae.blockChannels = [32, 32]
        vae.layersPerBlock = 1
        vae.normalizationGroups = 8
        let net = NFKMLXStableNormalNet(unet: unet, vae: vae)
        net.train(false)
        return net
    }

    func testYOSOIsMarigoldGeometryOverFourChannels() {
        let yoso = NFKMLXSDUNetConfiguration.yoso, marigold = NFKMLXSDUNetConfiguration.marigold
        XCTAssertEqual(yoso.inputChannels, 4)
        XCTAssertEqual(yoso.outputChannels, 4)
        XCTAssertEqual(yoso.blockChannels, marigold.blockChannels)
        XCTAssertEqual(yoso.attentionHeads, [5, 10, 20, 20])
        XCTAssertEqual(yoso.crossAttentionDimensions, 1024)
        XCTAssertTrue(yoso.usesLinearProjection)
    }

    func testControlNetVAEHoldsTheUNetsEncoderHalfUnderItsKeys() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXSDUNetConfiguration.yoso
        let controlnet = Set(NFKMLXSDControlNetVAE(configuration: configuration).parameters().flattened().map(\.0))
        let unet = Set(NFKMLXSDUNet(configuration: configuration).parameters().flattened().map(\.0))
        let encoderHalf = unet.filter { name in
            ["time_embedding.", "conv_in.", "down_blocks.", "mid_block."].contains { name.hasPrefix($0) }
        }
        XCTAssertEqual(controlnet, encoderHalf)
    }

    func testResidualsNumberTheSkipsAtTheirShapes() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let latent = MLXRandom.normal([1, 16, 16, 4])
        let residuals = net.residuals(latent)
        // conv_in, one resnet per level, and the first level's downsampler.
        XCTAssertEqual(residuals.down.map(\.shape), [[1, 16, 16, 32], [1, 16, 16, 32], [1, 8, 8, 32], [1, 8, 8, 64]])
        XCTAssertEqual(residuals.mid.shape, [1, 8, 8, 64])
        XCTAssertEqual(net.predictedLatent(latent, residuals: residuals).shape, [1, 16, 16, 4])
    }

    func testZeroResidualsLeaveTheUNetUnchanged() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let latent = MLXRandom.normal([1, 16, 16, 4])
        let shapes = net.residuals(latent)
        let zeros = NFKSDControlResiduals(down: shapes.down.map { MLXArray.zeros($0.shape) }, mid: MLXArray.zeros(shapes.mid.shape))
        let context = net.context, timestep = MLXArray([NFKMLXStableNormalNet.timestep])
        let plain = net.pipeline.unet(latent, timestep: timestep, context: context)
        let controlled = net.pipeline.unet(latent, timestep: timestep, context: context, controlResiduals: zeros)
        XCTAssertEqual(floats(plain), floats(controlled))
        XCTAssertNotEqual(floats(plain), floats(net.predictedLatent(latent, residuals: shapes)))
    }

    func testPredictorRunsAnRGBAImageBackToItsSize() throws {
        try requireMLXRuntime()
        let model = NFKMLXStableNormalModel(net: tinyNet(), processingResolution: 64)
        let width = 96, height = 64
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for pixel in 0 ..< width * height {
            rgba[pixel * 4] = UInt8(pixel % 251)
            rgba[pixel * 4 + 1] = UInt8(pixel % 13 * 19)
            rgba[pixel * 4 + 2] = 90
            rgba[pixel * 4 + 3] = pixel % width < width / 2 ? 255 : 0
        }
        let stages = try model.stages(rgba: rgba, width: width, height: height, resolution: 64)
        XCTAssertEqual(stages.processingWidth, 64)
        XCTAssertEqual(stages.processingHeight, 64)
        XCTAssertEqual(stages.pixels.shape, [1, 64, 64, 3])
        XCTAssertEqual(stages.decoded.shape, [1, 64, 64, 3])
        XCTAssertEqual(stages.normalImage.count, width * height * 3)
        let lengths = stride(from: 0, to: stages.prediction.count, by: 3).map { index -> Float in
            let p = stages.prediction
            return (p[index] * p[index] + p[index + 1] * p[index + 1] + p[index + 2] * p[index + 2]).squareRoot()
        }
        XCTAssertLessThan(lengths.map { abs($0 - 1) }.max() ?? 1, 1e-5)
        // The transparent right half lies outside the mask, so its map is white.
        XCTAssertEqual(stages.mask[63], 0)
        XCTAssertEqual(Array(stages.map[(64 - 1) * 3 ..< 64 * 3]), [255, 255, 255])
        XCTAssertThrowsError(try model.stages(rgba: rgba, width: width, height: height, resolution: 128))
    }

    // MARK: - Byte stages

    func testProcessingSizeFollowsResizeImage() {
        // Expected sizes from the reference's `resize_image` (NumPy rounding, half to even).
        let cases: [((Int, Int), (Int, Int))] = [((1024, 1024), (768, 768)), ((1024, 683), (768, 512)),
                                                 ((500, 300), (768, 448)), ((1200, 250), (768, 128)),
                                                 ((300, 500), (448, 768)), ((1920, 1080), (768, 448))]
        for (input, expected) in cases {
            let size = NFKMLXStableNormalModel.processingSize(width: input.0, height: input.1, resolution: 768)
            XCTAssertEqual(size.width, expected.0, "\(input)")
            XCTAssertEqual(size.height, expected.1, "\(input)")
        }
    }

    func testNearestResizePicksPillowsSourcePixels() {
        // Pillow 11's `resize(…, NEAREST)` of a row holding its own indices.
        let cases: [(Int, Int, [UInt8])] = [(7, 5, [0, 2, 3, 4, 6]), (5, 7, [0, 1, 1, 2, 3, 3, 4]),
                                            (3, 8, [0, 0, 0, 1, 1, 2, 2, 2])]
        for (inSize, outSize, expected) in cases {
            let row = (0 ..< inSize).map { UInt8($0) }
            XCTAssertEqual(NFKMLXStableNormalModel.nearest(row, width: inSize, height: 1, outWidth: outSize, outHeight: 1),
                           expected, "\(inSize) → \(outSize)")
        }
        // At 1024 → 768 the running sum at x = 4 is just under 6, where `(x + 0.5) · scale` is 6.
        let wide = (0 ..< 1024).map { UInt8($0 % 256) }
        let picked = NFKMLXStableNormalModel.nearest(wide, width: 1024, height: 1, outWidth: 768, outHeight: 1)
        XCTAssertEqual(Array(picked.prefix(12)), [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 14, 15])
    }

    func testCompositeAndMapRoundAsPillowAndNumPy() {
        let rgba: [UInt8] = [10, 200, 30, 0, 10, 200, 30, 255, 100, 50, 250, 128, 0, 0, 0, 1]
        let composite = NFKMLXStableNormalModel.rgb(NFKMLXImageBridge.compositedOnWhite(rgba: rgba))
        XCTAssertEqual(composite, [255, 255, 255, 10, 200, 30, 177, 152, 252, 254, 254, 254])
        XCTAssertEqual(NFKMLXStableNormalModel.byteMap([-1.5, -1, -0.2, 0, 0.3, 0.999, 1, 2]), [0, 0, 102, 127, 165, 254, 255, 255])
    }

    func testPredictorByteStagesMatchTheReferenceBitForBit() throws {
        try requireMLXRuntime()
        let record = try record()
        let prepared = try tensor(record, "prepared_image")
        let (height, width) = (prepared.dim(0), prepared.dim(1))
        let rgba = bytes(prepared)
        let composite = NFKMLXStableNormalModel.rgb(NFKMLXImageBridge.compositedOnWhite(rgba: rgba))
        XCTAssertEqual(composite, bytes(try tensor(record, "normal.composite")), "composite")

        let size = NFKMLXStableNormalModel.processingSize(width: width, height: height, resolution: 768)
        let resized = NFKMLXPillowLanczos.resize(composite, width: width, height: height, outWidth: size.width, outHeight: size.height)
        let pixels = NFKMLXStableNormalModel.pixels(resized, width: size.width, height: size.height)
        XCTAssertEqual(floats(pixels), floats(channelsLast(try tensor(record, "normal.pixels"))), "pixels")

        let alpha = (0 ..< width * height).map { rgba[$0 * 4 + 3] > 128 ? UInt8(1) : 0 }
        let mask = NFKMLXStableNormalModel.nearest(alpha, width: width, height: height, outWidth: size.width, outHeight: size.height)
        XCTAssertEqual(mask, bytes(try tensor(record, "normal.mask")), "mask")

        let masked = NFKMLXStableNormalModel.masked(floats(try tensor(record, "normal.prediction")), mask: mask)
        XCTAssertEqual(masked, floats(try tensor(record, "normal.masked")), "masked")
        let map = NFKMLXStableNormalModel.byteMap(masked)
        XCTAssertEqual(map, bytes(try tensor(record, "normal.map")), "map")
        let normalImage = NFKMLXPillowLanczos.resize(map, width: size.width, height: size.height, outWidth: width, outHeight: height)
        XCTAssertEqual(normalImage, bytes(try tensor(record, "normal_image")), "normal image")
    }

    // MARK: - Parity

    func testNormalStageMatchesTheReferenceSeamBySeam() throws {
        try requireMLXRuntime()
        let record = try record()
        let floors = try MLX.loadArrays(url: try envPath("IK_PARITY_HI3DGEN_FLOOR_NORMAL"))
        let model = try NFKMLXStableNormalModel.load(directoryURL: try envPath("IK_VAL_YOSO_NORMAL"))
        let net = model.net
        let prepared = try tensor(record, "prepared_image")
        let referenceDown = try (0 ..< 12).map { channelsLast(try tensor(record, "normal.controlnet.down\($0)")) }
        let referenceMid = channelsLast(try tensor(record, "normal.controlnet.mid"))
        let referenceLatent = channelsLast(try tensor(record, "normal.image_latent"))
        let referencePredicted = channelsLast(try tensor(record, "normal.unet_out"))
        let referenceDecoded = channelsLast(try tensor(record, "normal.decoded"))
        let referencePrediction = try tensor(record, "normal.prediction")

        // Each network from the reference's own input, so each seam measures one stage.
        assertWithinFloor(net.imageLatent(channelsLast(try tensor(record, "normal.pixels"))), floors, record,
                          "normal.image_latent", recorded: "normal.image_latent", label: "teacher-forced latent")
        let residuals = net.residuals(referenceLatent)
        for (index, residual) in residuals.down.enumerated() {
            let seam = "normal.controlnet.down\(index)"
            assertWithinFloor(residual, floors, record, seam, recorded: seam, label: "teacher-forced down\(index)")
        }
        assertWithinFloor(residuals.mid, floors, record, "normal.controlnet.mid", recorded: "normal.controlnet.mid",
                          label: "teacher-forced mid")
        let forced = NFKSDControlResiduals(down: referenceDown, mid: referenceMid)
        assertWithinFloor(net.predictedLatent(referenceLatent, residuals: forced), floors, record, "normal.unet_out",
                          recorded: "normal.unet_out", label: "teacher-forced UNet output")
        assertWithinFloor(net.decoded(referencePredicted), floors, record, "normal.decoded", recorded: "normal.decoded",
                          label: "teacher-forced decoded")
        assertCosine(NFKMLXStableNormalNet.normalized(referenceDecoded), referencePrediction, "teacher-forced prediction",
                     bar: Self.renormalizationBar)

        // The predictor end to end from the prepared image, against the float64 chain.
        let stages = try model.stages(rgba: bytes(prepared), width: prepared.dim(1), height: prepared.dim(0))
        XCTAssertEqual(stages.composite, bytes(try tensor(record, "normal.composite")), "composite")
        XCTAssertEqual(stages.mask, bytes(try tensor(record, "normal.mask")), "mask")
        XCTAssertEqual(floats(stages.pixels), floats(channelsLast(try tensor(record, "normal.pixels"))), "pixels")
        assertWithinFloor(stages.latent, floors, record, "normal.chained.image_latent", recorded: "normal.image_latent",
                          label: "latent")
        for (index, residual) in stages.residuals.down.enumerated() {
            assertWithinFloor(residual, floors, record, "normal.chained.controlnet.down\(index)",
                              recorded: "normal.controlnet.down\(index)", label: "down\(index)")
        }
        assertWithinFloor(stages.residuals.mid, floors, record, "normal.chained.controlnet.mid",
                          recorded: "normal.controlnet.mid", label: "mid")
        assertWithinFloor(stages.predictedLatent, floors, record, "normal.chained.unet_out", recorded: "normal.unet_out",
                          label: "UNet output")
        assertWithinFloor(stages.decoded, floors, record, "normal.chained.decoded", recorded: "normal.decoded",
                          label: "decoded")

        let map = byteDifferences(stages.map, bytes(try tensor(record, "normal.map")))
        report("StableNormal map: \(map.count) of \(stages.map.count) bytes differ, the largest by \(map.largest)")
        XCTAssertLessThanOrEqual(Double(map.count), Self.mapByteShare * Double(stages.map.count), "map")
        XCTAssertLessThanOrEqual(map.largest, 1, "map")
        let image = byteDifferences(stages.normalImage, bytes(try tensor(record, "normal_image")))
        report("StableNormal normal image: \(image.count) of \(stages.normalImage.count) bytes differ, the largest by \(image.largest)")
        XCTAssertLessThanOrEqual(Double(image.count), Self.normalImageByteShare * Double(stages.normalImage.count), "normal image")
        XCTAssertLessThanOrEqual(image.largest, 1, "normal image")
    }

    func testBackendReturnsTheNormalMapAtTheInputSize() throws {
        try requireMLXRuntime()
        let record = try record()
        let backend = try NFKMLXStableNormal.backend(directoryURL: try envPath("IK_VAL_YOSO_NORMAL"))
        XCTAssertEqual(backend.backendIdentifier, "stablenormal-turbo")
        let prepared = try tensor(record, "prepared_image")
        let (height, width) = (prepared.dim(0), prepared.dim(1))
        let input = try NFKMLXImageBridge.cgImage(rgba: bytes(prepared), width: width, height: height, options: NFKMLXImageOptions())
        let request = NFKInferenceRequest(inputs: [NFKInputImage: input])
        let result = try backend.runInference(for: request)
        let output = try XCTUnwrap(result.output(forKey: NFKOutputImage))
        let read = try NFKMLXImageBridge.rgbaBytes(from: output, colorSpace: CGColorSpaceCreateDeviceRGB())
        XCTAssertEqual(read.width, width)
        XCTAssertEqual(read.height, height)
        let rgb = NFKMLXStableNormalModel.rgb(read.bytes)
        let differences = byteDifferences(rgb, bytes(try tensor(record, "normal_image")))
        report("StableNormal backend: \(differences.count) of \(rgb.count) bytes differ, the largest by \(differences.largest)")
        XCTAssertLessThanOrEqual(Double(differences.count), Self.normalImageByteShare * Double(rgb.count))
        XCTAssertLessThanOrEqual(differences.largest, 1)
    }
}
