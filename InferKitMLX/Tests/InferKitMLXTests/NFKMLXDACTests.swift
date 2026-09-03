//
//  NFKMLXDACTests.swift
//  InferKitMLXTests
//
//  The Descript Audio Codec. Encode/decode and the backend evaluate MLX arrays, so they skip under
//  `swift test` and run under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXDACTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private func net() -> NFKMLXDACNet { NFKMLXDAC.makeNet(.tiny) }

    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let names = Set(net().parameters().flattened().map(\.0))
        for expected in ["encoder.conv_in.weight", "encoder.blocks.0.conv.weight",
                         "encoder.blocks.0.res_unit1.conv1.weight", "encoder.conv_out.weight",
                         "quantizer.quantizers.0.in_proj.weight", "quantizer.quantizers.0.out_proj.weight",
                         "quantizer.quantizers.0.codebook.weight",
                         "decoder.conv_in.weight", "decoder.blocks.0.conv_t1.conv.weight",
                         "decoder.conv_out.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    func testEncodeProducesOneCodePerCodebookPerFrameInRange() throws {
        try requireMLXRuntime()
        let codec = try NFKMLXDAC.codec(configuration: .tiny, weightsURL: nil)
        let codes = codec.encode(Self.floats(samples: 4096))               // hop 8 → 512 frames
        XCTAssertEqual(codes.count, NFKMLXDACConfiguration.tiny.codebooks, "one stream per codebook")
        XCTAssertEqual(codes[0].count, 512, "one code per frame")
        for stream in codes {
            for code in stream {
                XCTAssertGreaterThanOrEqual(code, 0)
                XCTAssertLessThan(code, NFKMLXDACConfiguration.tiny.codebookSize, "a code is a codebook index")
            }
        }
    }

    func testDecodeReconstructsAWaveformOfTheExpectedLength() throws {
        try requireMLXRuntime()
        let codec = try NFKMLXDAC.codec(configuration: .tiny, weightsURL: nil)
        let codes = codec.encode(Self.floats(samples: 4096))
        let audio = codec.decode(codes)
        XCTAssertEqual(audio.count, 512 * NFKMLXDACConfiguration.tiny.hopLength, "frames × hop samples")
        for sample in audio {
            XCTAssertGreaterThanOrEqual(sample, -1.0000001)
            XCTAssertLessThanOrEqual(sample, 1.0000001, "tanh keeps the waveform in range")
        }
    }

    func testEncodingIsDeterministic() throws {
        try requireMLXRuntime()
        let codec = try NFKMLXDAC.codec(configuration: .tiny, weightsURL: nil)
        let samples = Self.floats(samples: 2048)
        XCTAssertEqual(codec.encode(samples), codec.encode(samples), "the same clip codes the same way")
    }

    func testTheBackendReconstructsAudio() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXDAC.backend(weightsURL: nil)
        let wave = NFKMLXWaveFile.data(samples: Self.floats(samples: 4096), sampleRate: 44100)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave]))
        XCTAssertNotNil(result.output(forKey: NFKOutputAudio), "the reconstructed clip is returned")
    }

    static func floats(samples: Int) -> [Float] {
        (0 ..< samples).map { sinf(Float($0) * 0.05) * 0.4 }
    }
}
