//
//  NFKMLXCMGANTests.swift
//  InferKitMLXTests
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXCMGANTests: XCTestCase {

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

    // The generator keeps the compressed spectrum's shape and produces finite values.
    func testTheGeneratorKeepsTheSpectrumShape() throws {
        try requireMLXRuntime()
        let net = NFKMLXCMGAN.makeNet()
        let compressed = MLXRandom.normal([1, 20, 201, 2])
        let enhanced = net(compressed)
        eval(enhanced)
        XCTAssertEqual(enhanced.shape, [1, 20, 201, 2])
        XCTAssertTrue(enhanced.asArray(Float.self).allSatisfy(\.isFinite))
    }

    // The released checkpoint's names land on the module: every key maps and every mapped shape fits.
    func testTheReleasedKeysMapOntoTheModule() throws {
        try requireMLXRuntime()
        let mapped = ["TSCB_3.freq_conformer.attn.fn.rel_pos_emb.weight", "dense_encoder.dilated_dense.conv4.weight",
                      "mask_decoder.dense_block.prelu2.weight", "dense_encoder.conv_1.2.weight",
                      "TSCB_1.time_conformer.conv.net.4.conv.weight", "complex_decoder.conv.bias"]
            .map { NFKMLXCMGAN.remapReferenceKey($0) }
        XCTAssertEqual(mapped, ["tscb.2.freq_conformer.attn.fn.rel_pos_emb.weight", "dense_encoder.dilated_dense.layers.3.conv.weight",
                                "mask_decoder.dense_block.layers.1.prelu.weight", "dense_encoder.conv_1.2.weight",
                                "tscb.0.time_conformer.conv.net.4.conv.weight", "complex_decoder.conv.bias"])
        XCTAssertNil(NFKMLXCMGAN.remapReferenceKey("TSCB_1.time_conformer.conv.net.5.num_batches_tracked"))
    }

    // Seam by seam against the released TSCNet on the repository's own noisy clip: the compressed
    // spectrum, the dense encoder, each TSCB, the mask, the complex residual, the final parts, and the
    // enhanced waveform through the whole evaluation path.
    func testCMGANMatchesTheReference() throws {
        try requireMLXRuntime()
        let config = NFKMLXValidationConfig.environment
        guard let recordPath = config["IK_PARITY_CMGAN"], let directory = config["IK_VAL_CMGAN"] else {
            throw XCTSkip("set IK_PARITY_CMGAN and IK_VAL_CMGAN (run_reference.py cmgan)")
        }
        let record = try loadArrays(url: URL(fileURLWithPath: recordPath))
        let waveform = try XCTUnwrap(record["waveform"]).asArray(Float.self)
        let net = NFKMLXCMGAN.makeNet()
        try NFKMLXCMGAN.loadWeights(into: net, from: URL(fileURLWithPath: directory).appendingPathComponent("ckpt"))
        let configuration = net.configuration

        let (padded, scale) = NFKMLXCMGANBackend.prepared(waveform, configuration: configuration)
        XCTAssertEqual(scale, try XCTUnwrap(record["scale"]).item(Float.self), accuracy: 1e-5)
        XCTAssertEqual(padded.count, try XCTUnwrap(record["padded"]).dim(0))
        let compressed = NFKMLXCMGANBackend.compressed(padded, configuration: configuration)
        let referenceCompressed = try XCTUnwrap(record["compressed"]).transposed(1, 2, 0).expandedDimensions(axis: 0)
        eval(compressed)
        XCTAssertEqual(compressed.shape, referenceCompressed.shape)
        var seams = [("compressed", cosine(compressed, referenceCompressed))]

        // The network on the reference's own compressed spectrum, stage by stage.
        let outputs = net.blockOutputs(NFKMLXCMGANNet.encoderInput(referenceCompressed))
        for (name, output) in zip(["encoder", "tscb1", "tscb2", "tscb3", "tscb4"], outputs) {
            let reference = try XCTUnwrap(record[name]).transposed(1, 2, 0).expandedDimensions(axis: 0)
            eval(output)
            XCTAssertEqual(output.shape, reference.shape, name)
            seams.append((name, cosine(output, reference)))
        }
        let (_, mask, residual) = net.stages(referenceCompressed)
        seams.append(("mask", cosine(mask, try XCTUnwrap(record["mask"]).transposed(1, 2, 0).expandedDimensions(axis: 0))))
        seams.append(("complex", cosine(residual, try XCTUnwrap(record["complex"]).transposed(1, 2, 0).expandedDimensions(axis: 0))))
        let final = net(referenceCompressed)
        seams.append(("final_real", cosine(final[0..., 0..., 0..., 0], try XCTUnwrap(record["final_real"])[0])))
        seams.append(("final_imag", cosine(final[0..., 0..., 0..., 1], try XCTUnwrap(record["final_imag"])[0])))

        let enhanced = NFKMLXCMGANBackend.enhance(waveform, net: net)
        let reference = try XCTUnwrap(record["output"]).reshaped([1, -1])
        eval(enhanced)
        XCTAssertEqual(enhanced.shape, reference.shape)
        seams.append(("enhanced", cosine(enhanced, reference)))
        print("VALIDATION PARITY cmgan: " + seams.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
        for (name, value) in seams { XCTAssertGreaterThan(value, 0.999, name) }
    }
}
