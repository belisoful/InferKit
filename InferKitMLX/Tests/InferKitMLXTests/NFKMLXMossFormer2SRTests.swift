//
//  NFKMLXMossFormer2SRTests.swift
//  InferKitMLXTests
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXMossFormer2SRTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for (x, y) in zip(a, b) { dot += Double(x) * Double(y); na += Double(x) * Double(x); nb += Double(y) * Double(y) }
        return dot / (na.squareRoot() * nb.squareRoot())
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.reshaped([-1]).asType(.float32), y = b.reshaped([-1]).asType(.float32)
        let value = (x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()))
        eval(value)
        return Double(value.item(Float.self))
    }

    // Butterworth coefficients against scipy's `butter(4, 0.25, 'low')` and `butter(4, 0.25, 'high')`.
    func testTheButterworthDesignMatchesScipy() {
        let low = NFKMossBandwidthSubstitution.butterworth(order: 4, normalizedCutoff: 0.25, highpass: false)
        let expectedLowB = [0.01020948, 0.04083792, 0.06125688, 0.04083792, 0.01020948]
        let expectedLowA = [1.0, -1.96842779, 1.73586071, -0.72447083, 0.1203896]
        for (x, y) in zip(low.b, expectedLowB) { XCTAssertEqual(x, y, accuracy: 1e-7) }
        for (x, y) in zip(low.a, expectedLowA) { XCTAssertEqual(x, y, accuracy: 1e-7) }
        let high = NFKMossBandwidthSubstitution.butterworth(order: 4, normalizedCutoff: 0.25, highpass: true)
        let expectedHighB = [0.34682181, -1.38728723, 2.08093085, -1.38728723, 0.34682181]
        for (x, y) in zip(high.b, expectedHighB) { XCTAssertEqual(x, y, accuracy: 1e-7) }
        for (x, y) in zip(high.a, expectedLowA) { XCTAssertEqual(x, y, accuracy: 1e-7) }
    }

    // The generator upsamples a mel by the product of its rates and stays inside tanh's range.
    func testTheGeneratorUpsamplesTheMel() throws {
        try requireMLXRuntime()
        let net = NFKMLXMossFormer2SRFactory.makeNet()
        let waveform = net.generator(MLXRandom.normal([1, 6, 80]))
        eval(waveform)
        XCTAssertEqual(waveform.shape, [1, 6 * 256, 1])
        XCTAssertLessThanOrEqual(abs(waveform).max().item(Float.self), 1)
    }

    // Seam by seam against ClearerVoice's own models on the released weights: the log-mel, the
    // backbone on the reference mel, the generator on the reference restored mel, the detected
    // bandwidth, the scipy substitution on the reference generator output, and the whole path.
    func testMossFormer2SRMatchesTheReference() throws {
        try requireMLXRuntime()
        let config = NFKMLXValidationConfig.environment
        guard let recordPath = config["IK_PARITY_MOSSFORMER2_SR"], let directory = config["IK_VAL_MOSSFORMER2_SR"] else {
            throw XCTSkip("set IK_PARITY_MOSSFORMER2_SR and IK_VAL_MOSSFORMER2_SR (run_reference.py mossformer2_sr)")
        }
        let record = try loadArrays(url: URL(fileURLWithPath: recordPath))
        let waveform = try XCTUnwrap(record["waveform"]).asArray(Float.self)
        let net = NFKMLXMossFormer2SRFactory.makeNet()
        let directoryURL = URL(fileURLWithPath: directory)
        try NFKMLXMossFormer2SRFactory.loadBackboneWeights(into: net.backbone, from: directoryURL.appendingPathComponent(NFKMLXMossFormer2SRFactory.backboneFile))
        try NFKMLXMossFormer2SRFactory.loadGeneratorWeights(into: net.generator, from: directoryURL.appendingPathComponent(NFKMLXMossFormer2SRFactory.generatorFile))

        let referenceMel = try XCTUnwrap(record["mel"]).transposed(1, 0).expandedDimensions(axis: 0)   // [1, T, 80]
        let mel = net.logMel(waveform)
        eval(mel)
        XCTAssertEqual(mel.shape, referenceMel.shape)
        var seams = [("mel", cosine(mel, referenceMel))]

        let referenceRestored = try XCTUnwrap(record["restored_mel"]).transposed(1, 0).expandedDimensions(axis: 0)
        let restored = net.backbone(referenceMel)
        eval(restored)
        seams.append(("backbone", cosine(restored, referenceRestored)))

        let referenceGenerated = try XCTUnwrap(record["generated"]).asArray(Float.self)
        let generated = net.generated(referenceRestored)
        XCTAssertEqual(generated.count, referenceGenerated.count)
        seams.append(("generator", cosine(generated, referenceGenerated)))

        let configuration = net.configuration
        let cutoff = NFKMossBandwidthSubstitution.detectedBandwidth(waveform, sampleRate: configuration.sampleRate,
                                                                    size: configuration.bandwidthDetectionSize,
                                                                    threshold: configuration.bandwidthEnergyThreshold)
        XCTAssertEqual(cutoff, Double(try XCTUnwrap(record["f_high"]).item(Float.self)), accuracy: 1e-3)

        let referenceOutput = try XCTUnwrap(record["output"]).asArray(Float.self)
        let substituted = NFKMossBandwidthSubstitution.substituted(low: waveform, high: referenceGenerated, configuration: configuration)
        XCTAssertEqual(substituted.count, referenceOutput.count)
        seams.append(("substitution", cosine(substituted, referenceOutput)))

        let enhanced = net.enhance(waveform)
        seams.append(("enhanced", cosine(enhanced, referenceOutput)))
        print("VALIDATION PARITY mossformer2-sr: cutoff \(cutoff) Hz, " + seams.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
        for (name, value) in seams { XCTAssertGreaterThan(value, 0.999, name) }
    }
}
