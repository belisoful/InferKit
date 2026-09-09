//
//  NFKMLXMetricGANPlusTests.swift
//  InferKitMLXTests
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXMetricGANPlusTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.reshaped([-1]).asType(.float32), y = b.reshaped([-1]).asType(.float32)
        let value = (x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()))
        eval(value)
        return Double(value.item(Float.self))
    }

    // The generator takes log-magnitude frames and returns a mask of the same shape, every value in
    // the learnable sigmoid's 0 … 1.2 range.
    func testTheGeneratorMasksTheFrames() throws {
        try requireMLXRuntime()
        let net = NFKMLXMetricGANPlus.makeNet()
        let features = MLXRandom.uniform(0 ..< 1, [1, 12, 257])
        let mask = net(features)
        eval(mask)
        XCTAssertEqual(mask.shape, [1, 12, 257])
        XCTAssertGreaterThanOrEqual(mask.min().item(Float.self), 0)
        XCTAssertLessThanOrEqual(mask.max().item(Float.self), 1.2)
    }

    // Seam by seam against speechbrain's own SpectralMaskEnhancement on the released weights and the
    // model card's noisy clip: the zero-padded Hamming STFT features, the BLSTM mask, and the peak-
    // normalized enhanced waveform.
    func testMetricGANPlusMatchesTheReference() throws {
        try requireMLXRuntime()
        let config = NFKMLXValidationConfig.environment
        guard let recordPath = config["IK_PARITY_METRICGAN"], let directory = config["IK_VAL_METRICGAN"] else {
            throw XCTSkip("set IK_PARITY_METRICGAN and IK_VAL_METRICGAN (run_reference.py metricgan)")
        }
        let record = try loadArrays(url: URL(fileURLWithPath: recordPath))
        let waveform = try XCTUnwrap(record["waveform"]).asArray(Float.self)
        let net = NFKMLXMetricGANPlus.makeNet()
        try NFKMLXMetricGANPlus.loadWeights(into: net,
                                            from: URL(fileURLWithPath: directory).appendingPathComponent("enhance_model.ckpt"))

        let (features, _) = NFKMLXMetricGANPlusBackend.features(waveform, configuration: net.configuration)
        let referenceFeatures = try XCTUnwrap(record["features"]).expandedDimensions(axis: 0)
        eval(features)
        XCTAssertEqual(features.shape, referenceFeatures.shape)
        let featureCosine = cosine(features, referenceFeatures)

        let mask = net(referenceFeatures)
        eval(mask)
        let maskCosine = cosine(mask, try XCTUnwrap(record["mask"]).expandedDimensions(axis: 0))

        let enhanced = NFKMLXMetricGANPlusBackend.enhance(waveform, net: net)
        eval(enhanced)
        let reference = try XCTUnwrap(record["output"]).reshaped([1, -1])
        XCTAssertEqual(enhanced.shape, reference.shape)
        let outputCosine = cosine(enhanced, reference)
        print("VALIDATION PARITY metricgan-plus: features \(featureCosine), mask \(maskCosine), enhanced \(outputCosine), "
              + "peak \(abs(enhanced).max().item(Float.self))")
        XCTAssertGreaterThan(featureCosine, 0.9999)
        XCTAssertGreaterThan(maskCosine, 0.9999)
        XCTAssertGreaterThan(outputCosine, 0.999)
    }
}
