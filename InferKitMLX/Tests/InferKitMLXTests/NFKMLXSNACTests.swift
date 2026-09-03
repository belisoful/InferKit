//
//  NFKMLXSNACTests.swift
//  InferKitMLXTests
//
//  SNAC, the multi-scale codec. Encode/decode and the backend evaluate MLX arrays, so they skip under
//  `swift test` and run under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXSNACTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private func net() -> NFKMLXSNACNet { NFKMLXSNAC.makeNet(.tiny) }

    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(net().parameters().flattened().map(\.0))
        for expected in ["encoder.conv_in.weight", "encoder.blocks.0.res_unit1.conv1.weight",
                         "encoder.conv_out.weight", "quantizer.quantizers.0.in_proj.weight",
                         "quantizer.quantizers.0.codebook.weight", "decoder.conv_in_dw.weight",
                         "decoder.conv_in_pw.weight", "decoder.blocks.0.conv_t1.conv.weight",
                         "decoder.blocks.0.noise.linear.weight", "decoder.conv_out.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testEncodeProducesMultiScaleCodeStreamsInRange() throws {
        try requireMLXRuntime()
        let codec = try NFKMLXSNAC.codec(configuration: .tiny, weightsURL: nil)
        let codes = codec.encode(Self.floats(samples: 4096))               // hop 8 → 512 frames
        // vqStrides [2, 1] → the coarse codebook emits half as many tokens as the fine one.
        XCTAssertEqual(codes.map(\.count), [256, 512], "each codebook codes at its own temporal rate")
        for stream in codes {
            for code in stream {
                XCTAssertGreaterThanOrEqual(code, 0)
                XCTAssertLessThan(code, NFKMLXSNACConfiguration.tiny.codebookSize)
            }
        }
    }

    func testDecodeReconstructsAWaveformOfTheExpectedLength() throws {
        try requireMLXRuntime()
        let codec = try NFKMLXSNAC.codec(configuration: .tiny, weightsURL: nil)
        let codes = codec.encode(Self.floats(samples: 4096))
        let audio = codec.decode(codes, deterministic: true)
        XCTAssertEqual(audio.count, 512 * NFKMLXSNACConfiguration.tiny.hopLength, "frames × hop samples")
        for sample in audio {
            XCTAssertGreaterThanOrEqual(sample, -1.0000001)
            XCTAssertLessThanOrEqual(sample, 1.0000001, "tanh keeps the waveform in range")
        }
    }

    func testDeterministicDecodeIsReproducibleAndNoiseChangesIt() throws {
        try requireMLXRuntime()
        let codec = try NFKMLXSNAC.codec(configuration: .tiny, weightsURL: nil)
        let codes = codec.encode(Self.floats(samples: 2048))
        XCTAssertEqual(codec.decode(codes, deterministic: true), codec.decode(codes, deterministic: true),
                       "the noise-off decode is reproducible")
        // The noise block perturbs the decode, so enabling it changes the output.
        XCTAssertNotEqual(codec.decode(codes, deterministic: true), codec.decode(codes, deterministic: false),
                          "the noise block contributes when enabled")
    }

    func testTheBackendReconstructsAudio() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXSNAC.backend(weightsURL: nil)
        let wave = NFKMLXWaveFile.data(samples: Self.floats(samples: 4096), sampleRate: 24000)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave]))
        XCTAssertNotNil(result.output(forKey: NFKOutputAudio))
    }

    static func floats(samples: Int) -> [Float] {
        (0 ..< samples).map { sinf(Float($0) * 0.05) * 0.4 }
    }
}
