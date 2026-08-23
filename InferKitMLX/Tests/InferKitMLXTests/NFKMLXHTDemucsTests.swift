//
//  NFKMLXHTDemucsTests.swift
//  InferKitMLXTests
//
//  Demucs v4: a spectrogram branch and a waveform branch joined by a cross-transformer. These evaluate
//  MLX arrays, so they skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXHTDemucsTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    func testSeparateReturnsFourStereoStemsAtInputLength() throws {
        try requireMLXRuntime()
        let net = NFKMLXHTDemucs.makeNet(Self.tinyConfiguration())
        let stems = net.separate(Self.samples(8192), padsToTrainingSegment: false)
        eval(stems)
        XCTAssertEqual(stems.shape, [4, 2, 8192], "four stereo stems at the input length")
    }

    func testTheDefaultConfigurationIsTheReleasedMusicModel() {
        let configuration = NFKMLXHTDemucsConfiguration.htdemucs
        XCTAssertEqual(configuration.sources, 4)
        XCTAssertEqual(configuration.depth, 4)
        XCTAssertEqual(configuration.nFFT, 4096)
        XCTAssertEqual(configuration.hop, 1024, "the transform hops a quarter of its window")
        XCTAssertEqual(configuration.bottomChannels, 512, "the branches meet at the transformer's width")
        XCTAssertEqual(configuration.transformerLayers, 5)
    }

    func testAShortClipRunsAtTheTrainingSegmentAndComesBackTrimmed() throws {
        try requireMLXRuntime()
        var configuration = Self.tinyConfiguration()
        configuration.trainingSamples = 16384
        let net = NFKMLXHTDemucs.makeNet(configuration)
        let stems = net.separate(Self.samples(8192))
        eval(stems)
        XCTAssertEqual(stems.shape, [4, 2, 8192], "the padding to the training segment is trimmed back")
    }

    func testTheTransformInvertsToTheSignalItStartedFrom() throws {
        try requireMLXRuntime()
        let signal = (0 ..< 4096).map { sinf(Float($0) * 0.05) + 0.3 * sinf(Float($0) * 0.31) }
        let spectrum = NFKHTDemucsSpectrum.transform([signal], nFFT: 256, hop: 64)
        let restored = NFKHTDemucsSpectrum.inverseTransform(spectrum, nFFT: 256, hop: 64,
                                                            length: signal.count)
        let error = zip(signal, restored[0]).map { abs($0 - $1) }.max() ?? 1
        XCTAssertLessThan(error, 1e-4, "the windowed overlap-add reconstructs its own input")
    }

    func testACheckpointRoundTripReproducesTheSeparation() throws {
        try requireMLXRuntime()
        let trained = NFKMLXHTDemucs.makeNet(Self.tinyConfiguration())
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened()
            .map { key, value -> (String, MLXArray) in
                // A transposed convolution's axes rotate the other way, and the channel samplers are
                // linear layers here where the reference stores 1×1 convolutions.
                if key.hasSuffix("conv_tr.weight") {
                    return (key, value.ndim == 4 ? value.transposed(3, 0, 1, 2) : value.transposed(2, 0, 1))
                }
                if key.hasPrefix("channel_") && value.ndim == 2 {
                    return (key, value.expandedDimensions(axis: 2))
                }
                if value.ndim == 4 { return (key, value.transposed(0, 3, 1, 2)) }
                if value.ndim == 3 { return (key, value.transposed(0, 2, 1)) }
                return (key, value)
            })
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("htdemucs-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = NFKMLXHTDemucs.makeNet(Self.tinyConfiguration())
        try NFKMLXHTDemucs.loadWeights(into: loaded, from: url)

        let input = Self.samples(8192)
        let expected = trained.separate(input, padsToTrainingSegment: false)
        let actual = loaded.separate(input, padsToTrainingSegment: false)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self))
    }

    func testTheRemapTranslatesTheReferenceKeys() {
        XCTAssertEqual(NFKMLXHTDemucs.remapReferenceKey("encoder.0.dconv.layers.1.3.weight"),
                       "encoder.0.dconv.layers.1.conv_out.weight")
        XCTAssertEqual(NFKMLXHTDemucs.remapReferenceKey("tdecoder.2.dconv.layers.0.6.scale"),
                       "tdecoder.2.dconv.layers.0.layer_scale.scale")
        XCTAssertEqual(NFKMLXHTDemucs.remapReferenceKey("crosstransformer.layers.3.cross_attn.in_proj_weight"),
                       "crosstransformer.cross_layers.1.attn.in_proj_weight")
        XCTAssertEqual(NFKMLXHTDemucs.remapReferenceKey("crosstransformer.layers_t.4.self_attn.out_proj.bias"),
                       "crosstransformer.self_layers_t.2.attn.out_proj.bias")
        XCTAssertEqual(NFKMLXHTDemucs.remapReferenceKey("freq_emb.embedding.weight"), "freq_emb.weight")
    }

    func testTheBackendWritesAWaveFilePerStem() throws {
        try requireMLXRuntime()
        let inputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("htdemucs-in-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: inputURL) }
        try NFKMLXWaveFile.write(samples: (0 ..< 8192).map { sinf(Float($0) * 0.1) },
                                 sampleRate: 44100, to: inputURL)

        var configuration = Self.tinyConfiguration()
        configuration.trainingSamples = 8192
        let backend = NFKMLXHTDemucsBackend(net: NFKMLXHTDemucs.makeNet(configuration))
        let request = NFKInferenceRequest(inputs: [NFKInputAudio: NFKAudioAsset(
            fileURL: inputURL, durationSeconds: 0, sampleRate: 44100, channelCount: 1)])
        let result = try backend.runInference(for: request)
        for name in ["drums", "bass", "other", "vocals"] {
            let stem = try XCTUnwrap(result.output(forKey: name) as? NFKAudioAsset, "\(name) stem")
            XCTAssertEqual(stem.channelCount, 2, "\(name) keeps the model's stereo channels")
            let url = try XCTUnwrap(stem.fileURL)
            defer { try? FileManager.default.removeItem(at: url) }
            XCTAssertNotNil(NFKMLXWaveFile.read(try Data(contentsOf: url)), "\(name) is a readable WAV")
        }
    }

    /// The released geometry at a size a test can run: the same two branches, cross-transformer, and
    /// four depths, over a short transform and narrow channels.
    static func tinyConfiguration() -> NFKMLXHTDemucsConfiguration {
        var configuration = NFKMLXHTDemucsConfiguration.htdemucs
        configuration.channels = 8
        configuration.nFFT = 1024
        configuration.bottomChannels = 32
        configuration.transformerHeads = 2
        configuration.transformerHidden = 64
        configuration.trainingSamples = 8192
        return configuration
    }

    static func samples(_ count: Int) -> MLXArray {
        let values = (0 ..< (2 * count)).map { Float(($0 * 13) % 100) / 100 - 0.5 }
        return values.withUnsafeBufferPointer { MLXArray($0, [2, count]) }
    }
}
