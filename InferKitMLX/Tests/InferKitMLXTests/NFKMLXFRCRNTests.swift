//
//  NFKMLXFRCRNTests.swift
//  InferKitMLXTests
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXFRCRNTests: XCTestCase {

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

    /// `[B, D, T, C]` real/imaginary → `[D, T, C, 2]` for the batch's first entry.
    private func packed(_ x: NFKFRCRNComplex) -> MLXArray {
        stacked([x.real, x.imaginary], axis: -1)[0]
    }

    /// The reference `[C, D, T, 2]` → `[D, T, C, 2]`.
    private func reference(_ record: [String: MLXArray], _ key: String) throws -> MLXArray {
        try XCTUnwrap(record[key]).transposed(1, 2, 0, 3)
    }

    // The network keeps the spectrum's shape through both UNets and produces finite values.
    func testTheMaskKeepsTheSpectrumShape() throws {
        try requireMLXRuntime()
        let net = NFKMLXFRCRN.makeNet()
        let spectrum = NFKFRCRNComplex(real: MLXRandom.normal([1, 321, 12, 1]), imaginary: MLXRandom.normal([1, 321, 12, 1]))
        let mask = net.mask(spectrum).mask
        eval(mask.real, mask.imaginary)
        XCTAssertEqual(mask.real.shape, [1, 321, 12, 1])
        XCTAssertTrue(mask.imaginary.asArray(Float.self).allSatisfy(\.isFinite))
    }

    // The decode padding rule and the flat-copy drop in the remap.
    func testTheDecodePaddingFollowsTheReferenceGrid() throws {
        let c = NFKMLXFRCRNConfiguration()
        XCTAssertEqual(NFKMLXFRCRNBackend.padded([Float](repeating: 1, count: 1000), configuration: c).count, 16_000)
        XCTAssertEqual(NFKMLXFRCRNBackend.padded([Float](repeating: 1, count: 20_000), configuration: c).count, 28_000)
        XCTAssertEqual(NFKMLXFRCRNBackend.padded([Float](repeating: 1, count: 35_184), configuration: c).count, 58_368)
        XCTAssertEqual(NFKMLXFRCRNBackend.padded([Float](repeating: 1, count: 40_000), configuration: c).count, 40_000)
        XCTAssertNil(NFKMLXFRCRN.remapReferenceKey("unet.encoder3.conv.conv_re.weight"))
        XCTAssertNil(NFKMLXFRCRN.remapReferenceKey("unet2.fsmn_dec0.fsmn_re_L1.linear.weight"))
        XCTAssertEqual(NFKMLXFRCRN.remapReferenceKey("unet.encoders.3.conv.conv_re.weight"), "unet.encoders.3.conv.conv_re.weight")
        XCTAssertEqual(NFKMLXFRCRN.remapReferenceKey("unet.fsmn.fsmn_re_L2.conv1.weight"), "unet.fsmn.fsmn_re_L2.conv1.weight")
    }

    // Seam by seam against the released DCCRN on the noisy clip: the conv-STFT spectrum, encoder 0
    // before and after its squeeze-excite, the bottleneck FSMN, decoder 0, the first UNet, the combined
    // mask, the masked spectrum, and the enhanced waveform through the whole decode path.
    func testFRCRNMatchesTheReference() throws {
        try requireMLXRuntime()
        let config = NFKMLXValidationConfig.environment
        guard let recordPath = config["IK_PARITY_FRCRN"], let checkpoint = config["IK_VAL_FRCRN"] else {
            throw XCTSkip("set IK_PARITY_FRCRN and IK_VAL_FRCRN (run_reference.py frcrn)")
        }
        let record = try loadArrays(url: URL(fileURLWithPath: recordPath))
        let waveform = try XCTUnwrap(record["waveform"]).asArray(Float.self)
        let net = NFKMLXFRCRN.makeNet()
        try NFKMLXFRCRN.loadWeights(into: net, from: URL(fileURLWithPath: checkpoint))
        let configuration = net.configuration

        let padded = NFKMLXFRCRNBackend.padded(waveform, configuration: configuration)
        XCTAssertEqual(padded.count, try XCTUnwrap(record["padded"]).dim(0))
        let spectrum = NFKMLXFRCRNBackend.spectrum(padded, configuration: configuration)
        let referenceSpectrum = try reference(record, "spec")
        eval(spectrum.real, spectrum.imaginary)
        XCTAssertEqual(packed(spectrum).shape, referenceSpectrum.shape)
        var seams = [("spectrum", cosine(packed(spectrum), referenceSpectrum))]

        // The network on the reference's own spectrum.
        let input = NFKFRCRNComplex(real: referenceSpectrum[0..., 0..., 0..., 0].expandedDimensions(axis: 0),
                                    imaginary: referenceSpectrum[0..., 0..., 0..., 1].expandedDimensions(axis: 0))
        let stages = net.first.stagesOutput(input)
        for (name, value, key) in [("encoder0", stages.encoder0, "encoder0"), ("se0", stages.excited0, "se0"),
                                   ("bottleneck", stages.bottleneck, "bottleneck"), ("decoder0", stages.decoder0, "decoder0"),
                                   ("unet1", stages.output, "unet1")] {
            let ref = try reference(record, key)
            eval(value.real, value.imaginary)
            XCTAssertEqual(packed(value).shape, ref.shape, name)
            seams.append((name, cosine(packed(value), ref)))
        }
        let (_, mask) = net.mask(input)
        seams.append(("mask", cosine(packed(mask), try reference(record, "mask"))))
        // apply_mask packs the masked spectrum as [2·bins, T]: real bins then imaginary bins.
        let masked = net.masked(input)
        let referenceMasked = try XCTUnwrap(record["est_spec"])                   // [2·bins, T]
        let bins = configuration.bins
        seams.append(("masked", cosine(concatenated([masked.real[0, 0..., 0..., 0], masked.imaginary[0, 0..., 0..., 0]], axis: 0),
                                       referenceMasked[0 ..< 2 * bins])))

        let enhanced = NFKMLXFRCRNBackend.enhance(waveform, net: net)
        let referenceWave = try XCTUnwrap(record["output"])
        eval(enhanced)
        XCTAssertEqual(enhanced.dim(1), waveform.count)
        seams.append(("enhanced", cosine(enhanced[0], referenceWave[0 ..< waveform.count])))
        print("VALIDATION PARITY frcrn: " + seams.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
        for (name, value) in seams { XCTAssertGreaterThan(value, 0.999, name) }
    }
}
