//
//  NFKMLXBigVGANTests.swift
//  InferKitMLXTests
//
//  BigVGAN v2 (nvidia/bigvgan_v2_24khz_100band_256x, MIT), the anti-aliased SnakeBeta vocoder shipped as
//  a standalone copy-synthesis model. The generator evaluates MLX arrays, so these skip without a Metal
//  library for MLX (see Tools/mlx-metallib.sh). The parity test is gated on the released generator plus
//  the recorded oracle (`IK_VAL_BIGVGAN` + `IK_PARITY_BIGVGAN`, from `run_reference.py bigvgan`).
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXBigVGANTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Float {
        eval(mine)
        let a = mine.asArray(Float.self), b = reference.asArray(Float.self)
        let n = min(a.count, b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrtf(na) * sqrtf(nb) + 1e-20)
    }

    /// The module keys the loader targets: `conv_pre`, the `ups` transposed convolutions, the AMP block
    /// convolutions and their anti-aliased SnakeBeta activations, and `conv_post`.
    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let net = NFKMLXBigVGANFactory.makeNet()
        let names = Set(net.parameters().flattened().map(\.0))
        for expected in ["conv_pre.weight", "ups.0.weight",
                         "resblocks.0.convs1.0.weight", "resblocks.0.convs2.0.weight",
                         "resblocks.0.activations.0.act.alpha", "resblocks.0.activations.0.act.beta",
                         "activation_post.act.alpha", "conv_post.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    /// PARITY: the released BigVGAN generator against the recorded oracle (mel → waveform), through the
    /// shipped factory, including the anti-aliased SnakeBeta activations and the kaiser-sinc resamplings
    /// this port recomputes.
    func testGeneratorParityOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let weightsPath = env["IK_VAL_BIGVGAN"], let recordPath = env["IK_PARITY_BIGVGAN"] else {
            throw XCTSkip("set IK_VAL_BIGVGAN (generator) and IK_PARITY_BIGVGAN (oracle record)")
        }
        let net = NFKMLXBigVGANFactory.makeNet()
        try net.loadWeights(from: URL(fileURLWithPath: weightsPath))
        let r = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let mel = r["mel"]!.transposed(1, 0).expandedDimensions(axis: 0)   // [100, T] → [1, T, 100]
        let wav = net(mel)                                                 // [1, T·256, 1]
        XCTAssertGreaterThan(cosine(wav[0][0..., 0], r["output"]!), 0.9999, "BigVGAN waveform diverges")
    }

    /// The standalone copy-synthesis backend resynthesizes a clip through its own mel front end and the
    /// generator, returning a 24 kHz clip clamped to `[-1, 1]`.
    func testTheBackendResynthesizesAudio() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXBigVGANFactory.backend(weightsURL: nil)
        let wave = NFKMLXWaveFile.data(samples: (0 ..< 6144).map { sinf(Float($0) * 0.05) * 0.4 }, sampleRate: 24000)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wave]))
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputAudio) as? NFKAudioAsset,
                                  "the resynthesized clip is returned")
        XCTAssertEqual(asset.sampleRate, 24000)
        let url = try XCTUnwrap(asset.fileURL)
        let (samples, _) = try XCTUnwrap(NFKMLXWaveFile.read(Data(contentsOf: url)))
        XCTAssertGreaterThan(samples.count, 0, "the clip is non-empty")
        for sample in samples {
            XCTAssertGreaterThanOrEqual(sample, -1.0000001)
            XCTAssertLessThanOrEqual(sample, 1.0000001, "the generator clamps the waveform to [-1, 1]")
        }
    }

    /// The registry builds a ready backend under the model name, declaring the audio input it reads.
    func testTheRegistryPathBuildsAReadyBackend() throws {
        try requireMLXRuntime()
        NFKMLXBigVGANFactory.register()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXBigVGANFactory.modelName))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXBigVGANFactory.modelName, weightsURL: nil)
        XCTAssertTrue(backend.isReady)
        XCTAssertEqual(backend.backendIdentifier, NFKMLXBigVGANFactory.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputAudio])
    }
}
