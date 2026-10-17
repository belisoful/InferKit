//
//  NFKMLXSDAutoencoderDecodeStagingTests.swift
//  InferKitMLXTests
//
//  The SD-family autoencoder's decode: its one-graph peak estimate, the choice between one graph and a staged
//  decode, and the staged decode's bits, which equal the one-graph decode's.
//

import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXSDAutoencoderDecodeStagingTests: XCTestCase {

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    func testDecoderGeometryNamesTheWidestOutputSizeMap() {
        let sd = NFKSDVAEDecoder(.stableDiffusion)
        XCTAssertEqual(sd.outputScale, 8)
        XCTAssertEqual(sd.widestOutputChannels, 256)
        let flux2Small = NFKSDVAEDecoder(.flux2SmallDecoder)
        let widths = NFKMLXSDVAEConfiguration.flux2SmallDecoder.decoderBlockChannels
            ?? NFKMLXSDVAEConfiguration.flux2SmallDecoder.blockChannels
        XCTAssertEqual(flux2Small.widestOutputChannels, max(widths[0], widths[1]))
    }

    func testWholeGraphPeakEstimateCoversEachMeasuredPeak() {
        // (output side, widest output-size channels, measured one-graph peak in bytes), under the standing limits.
        let measured: [(Int, Int, Int)] = [
            (512, 256, 3_770_000_000), (768, 256, 7_610_000_000), (1024, 256, 13_500_000_000),
            (512, 256, 4_720_000_000), (768, 256, 7_550_000_000), (1024, 256, 12_370_000_000),
            (768, 192, 5_700_000_000),
        ]
        for (side, channels, peak) in measured {
            let estimate = NFKSDDecodeStaging.estimatedWholeGraphPeak(images: 1, outputPixels: side * side,
                                                                      widestOutputChannels: channels)
            XCTAssertGreaterThanOrEqual(estimate, peak, "\(side)² over \(channels) channels")
            XCTAssertLessThan(estimate - peak, 2_600_000_000, "\(side)² over \(channels) channels")
        }
        XCTAssertEqual(NFKSDDecodeStaging.estimatedWholeGraphPeak(images: 2, outputPixels: 1024 * 1024,
                                                                  widestOutputChannels: 256),
                       2 * (1_400_000_000 + 1024 * 1024 * 256 * 504 / 10))
    }

    func testDecodeStagesOnlyPastTheBudget() {
        XCTAssertNil(NFKSDDecodeStaging.evaluationBytes(estimatedPeak: 14_000_000_000, budget: 22_000_000_000))
        XCTAssertNil(NFKSDDecodeStaging.evaluationBytes(estimatedPeak: 14_000_000_000, budget: 14_000_000_000))
        XCTAssertEqual(NFKSDDecodeStaging.evaluationBytes(estimatedPeak: 14_000_000_000, budget: 10_000_000_000),
                       1 << 28)
        XCTAssertEqual(NFKSDDecodeStaging.evaluationBytes(estimatedPeak: 1, budget: -1), 1 << 28)
    }

    func testStagedDecodeGivesTheOneGraphBits() throws {
        try requireMLXRuntime()
        MLXRandom.seed(0)
        var configuration = NFKMLXSDVAEConfiguration()
        configuration.blockChannels = [32, 64]
        configuration.layersPerBlock = 1
        let vae = NFKMLXSDAutoencoder(configuration: configuration)
        let latent = MLXRandom.normal([1, 8, 8, 4])
        for dtype in [DType.float32, .bfloat16] {
            vae.update(parameters: vae.parameters().mapValues { $0.asType(dtype) })
            let x = latent.asType(dtype)
            let whole = vae.decoder.decoded(x, evaluationBytes: nil)
            let staged = vae.decoder.decoded(x, evaluationBytes: 0)
            eval(whole, staged)
            XCTAssertEqual(whole.shape, [1, 16, 16, 3])
            XCTAssertTrue(isFinite(staged).all().item(Bool.self))
            XCTAssertTrue(arrayEqual(whole, staged).item(Bool.self), "\(dtype)")
        }
    }
}
