//
//  NFKMLXNUWave2Tests.swift
//  InferKitMLXTests
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXNUWave2Tests: XCTestCase {

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

    // The normalized STFT pair inverts exactly for a length that is a multiple of the hop.
    func testTheNormalizedSpectrumRoundTrips() throws {
        try requireMLXRuntime()
        let spectrum = NFKNUWaveSpectrum(nFFT: 1024, hop: 256)
        let signal = MLXRandom.normal([3, 4096])
        let (real, imaginary) = spectrum.transform(signal)
        let restored = spectrum.inverse(real: real, imaginary: imaginary)
        eval(restored)
        XCTAssertEqual(restored.shape, [3, 4096])
        XCTAssertGreaterThan(cosine(restored, signal), 0.999999)
    }

    // The band marks the bins below the source's Nyquist with the reference's float arithmetic.
    func testTheBandFollowsTheSourceRate() {
        let c = NFKMLXNUWave2Configuration()
        let (narrowband, band) = NFKMLXNUWave2Backend.prepared([Float](repeating: 0.5, count: 16_000), sampleRate: 16_000, configuration: c)
        XCTAssertEqual(narrowband.count % 256, 0)
        XCTAssertEqual(band.reduce(0, +), 171)
        XCTAssertEqual(NFKMLXNUWave2Backend.prepared([0.5, 0.5], sampleRate: 24_000, configuration: c).band.reduce(0, +), 256)
    }

    // Seam by seam against the released Diffusion wrapper from the reference's own start noise: the
    // diffusion embedding, the first residual block, the step-0 noise prediction, every DDIM step, and
    // the clamped output.
    func testNUWave2MatchesTheReference() throws {
        try requireMLXRuntime()
        let config = NFKMLXValidationConfig.environment
        guard let recordPath = config["IK_PARITY_NUWAVE2"], let checkpoint = config["IK_VAL_NUWAVE2"] else {
            throw XCTSkip("set IK_PARITY_NUWAVE2 and IK_VAL_NUWAVE2 (run_reference.py nuwave2)")
        }
        let record = try loadArrays(url: URL(fileURLWithPath: recordPath))
        let net = NFKMLXNUWave2.makeNet()
        try NFKMLXNUWave2.loadWeights(into: net, from: URL(fileURLWithPath: checkpoint))
        let c = net.configuration
        let narrowband = try XCTUnwrap(record["waveform_low"]).reshaped([1, -1])
        let band = try XCTUnwrap(record["band"]).asType(.int32).reshaped([1, c.bins])
        let noise = try XCTUnwrap(record["noise"]).reshaped([1, -1])

        // The consumer path's band from the source rate matches the record's.
        let sourceRate = Int(try XCTUnwrap(record["source_rate"]).item(Float.self))
        let prepared = NFKMLXNUWave2Backend.prepared(try XCTUnwrap(record["source"]).asArray(Float.self), sampleRate: sourceRate, configuration: c)
        XCTAssertEqual(prepared.band.reduce(0, +), Int32(try XCTUnwrap(record["band"]).sum().item(Int64.self)))

        var seams = [(String, Double)]()
        let level = MLXArray([(c.logSNRMaximum - c.schedule[0]) / (c.logSNRMaximum - c.logSNRMinimum)])
        let embedded = net.embedding(level)
        eval(embedded)
        seams.append(("embedding", cosine(embedded, try XCTUnwrap(record["emb0"]))))
        let (predicted, first) = net.stages(noise, narrowband: narrowband, band: band, level: level)
        eval(predicted, first.residual, first.skip)
        seams.append(("layer0_x", cosine(first.residual, try XCTUnwrap(record["layer0_x"]).transposed(1, 0))))
        seams.append(("layer0_skip", cosine(first.skip, try XCTUnwrap(record["layer0_skip"]).transposed(1, 0))))
        seams.append(("eps0", cosine(predicted, try XCTUnwrap(record["eps0"]))))

        let (final, trajectory) = NFKMLXNUWave2Sampler.sample(net, narrowband: narrowband, band: band, noise: noise)
        for (i, signal) in trajectory.enumerated() {
            seams.append(("step\(i)", cosine(signal, try XCTUnwrap(record["step_\(i)"]))))
        }
        eval(final)
        seams.append(("output", cosine(final, try XCTUnwrap(record["output"]))))
        print("VALIDATION PARITY nuwave2: " + seams.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
        for (name, value) in seams { XCTAssertGreaterThan(value, 0.999, name) }
    }
}
