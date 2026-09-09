//
//  NFKMLXApolloTests.swift
//  InferKitMLXTests
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXApolloTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.reshaped([-1]).asType(.float32), y = b.reshaped([-1]).asType(.float32)
        let value = (x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()))
        eval(value)
        return Double(value.item(Float.self))
    }

    // The released geometry: 79 bands of 5 bins and a 47-bin remainder over 442 bins.
    func testTheBandSplitCoversTheSpectrum() {
        let c = NFKMLXApolloConfiguration()
        XCTAssertEqual(c.window, 882)
        XCTAssertEqual(c.bandWidths.count, 80)
        XCTAssertEqual(c.bandWidths.reduce(0, +), c.bins)
        XCTAssertEqual(c.bandWidths.last, 47)
        XCTAssertEqual(NFKMLXApollo.remapReferenceKey("BN.3.0.weight"), "BN.3.norm.weight")
        XCTAssertEqual(NFKMLXApollo.remapReferenceKey("output.79.1.bias"), "output.79.conv.bias")
        XCTAssertNil(NFKMLXApollo.remapReferenceKey("net.0.band_net.cos_freq"))
    }

    // The network returns a clip of the input's length for a length that is not a multiple of the hop.
    func testTheNetworkKeepsTheInputLength() throws {
        try requireMLXRuntime()
        let net = NFKMLXApollo.makeNet()
        let output = net(MLXRandom.normal([1, 10_000]))
        eval(output)
        XCTAssertEqual(output.shape, [1, 10_000])
        XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
    }

    // Seam by seam against the released Apollo on the repository's own clip: the band features, band
    // 0's bottleneck, the first and last band-sequence layers, band 0's head, and the restored waveform.
    func testApolloMatchesTheReference() throws {
        try requireMLXRuntime()
        let config = NFKMLXValidationConfig.environment
        guard let recordPath = config["IK_PARITY_APOLLO"], let checkpoint = config["IK_VAL_APOLLO"] else {
            throw XCTSkip("set IK_PARITY_APOLLO and IK_VAL_APOLLO (run_reference.py apollo)")
        }
        let record = try loadArrays(url: URL(fileURLWithPath: recordPath))
        let net = NFKMLXApollo.makeNet()
        try NFKMLXApollo.loadWeights(into: net, from: URL(fileURLWithPath: checkpoint))
        let waveform = try XCTUnwrap(record["waveform"]).reshaped([1, -1])

        let features = net.features(waveform)                                  // [1, 80, T, 256]
        eval(features)
        let referenceFeatures = try XCTUnwrap(record["features"]).transposed(0, 2, 1).expandedDimensions(axis: 0)
        XCTAssertEqual(features.shape, referenceFeatures.shape)
        var seams = [("features", cosine(features, referenceFeatures))]
        seams.append(("bn0", cosine(features[0..., 0], try XCTUnwrap(record["bn0"]).transposed(1, 0).expandedDimensions(axis: 0))))

        let outputs = net.layerOutputs(referenceFeatures)
        seams.append(("net0", cosine(outputs[0], try XCTUnwrap(record["net0"]).transposed(0, 2, 1).expandedDimensions(axis: 0))))
        let referenceFinal = try XCTUnwrap(record["net_final"]).transposed(0, 2, 1).expandedDimensions(axis: 0)
        seams.append(("net_final", cosine(outputs[outputs.count - 1], referenceFinal)))

        let head0 = net.heads[0](referenceFinal[0..., 0])
        eval(head0)
        seams.append(("head0", cosine(head0, try XCTUnwrap(record["head0"]).transposed(1, 0).expandedDimensions(axis: 0))))

        let restored = net(waveform)
        eval(restored)
        let referenceOutput = try XCTUnwrap(record["output"]).reshaped([1, -1])
        XCTAssertEqual(restored.shape, referenceOutput.shape)
        seams.append(("output", cosine(restored, referenceOutput)))
        print("VALIDATION PARITY apollo: " + seams.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
        for (name, value) in seams { XCTAssertGreaterThan(value, 0.999, name) }
    }
}
