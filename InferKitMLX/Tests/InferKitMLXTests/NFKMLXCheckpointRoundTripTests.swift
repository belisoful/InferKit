//
//  NFKMLXCheckpointRoundTripTests.swift
//  InferKitMLXTests
//
//  A model's own `loadWeights` has to read back what `NFKMLXWeights.save` writes, because a fine-tuned
//  checkpoint reloads through the model's existing `backendWith…weightsURL:` factory. The file records
//  its layout, and the loader skips its PyTorch transpose for a file already in the module's layout —
//  so a model that ignored the flag would transpose twice and load silently wrong weights.
//
//  The models here are the ones whose transpose is NOT the common `[out, in, kH, kW]` case, which is
//  where a double transpose does the most damage: SAM's `up1`/`up2` and the Colorizer's deconvolutions
//  use `transposed(1, 2, 3, 0)`, and Demucs carries 3-D Conv1d weights.
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXCheckpointRoundTripTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private func temporaryURL() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).safetensors")
    }

    /// Saves `source`, reloads through `load`, and asserts every parameter survives unchanged.
    private func assertRoundTrips(_ source: Module, into destination: Module,
                                  load: (Module, URL) throws -> Void,
                                  file: StaticString = #filePath, line: UInt = #line) throws {
        let url = temporaryURL()
        defer { try? FileManager.default.removeItem(at: url) }

        try NFKMLXWeights.save(source, to: url)
        XCTAssertFalse(try NFKMLXWeights.loadCheckpoint(url: url).needsConvTranspose,
                       "a saved checkpoint records the module's own layout", file: file, line: line)
        try load(destination, url)

        let before = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        let after = Dictionary(uniqueKeysWithValues: destination.parameters().flattened())
        XCTAssertEqual(before.count, after.count, "the same parameters come back", file: file, line: line)
        for (key, original) in before {
            guard let reloaded = after[key] else {
                XCTFail("\(key) is missing after the round trip", file: file, line: line); continue
            }
            XCTAssertEqual(original.shape, reloaded.shape, "\(key) keeps its shape", file: file, line: line)
            eval(original, reloaded)
            let worst = (original - reloaded).abs().max().item(Float.self)
            XCTAssertEqual(worst, 0, accuracy: 0, "\(key) reloads exactly", file: file, line: line)
        }
    }

    func testSAMRoundTripsThroughItsOwnLoader() throws {
        try requireMLXRuntime()
        let source = NFKMLXSAM.makeNet()
        let destination = NFKMLXSAM.makeNet()
        try assertRoundTrips(source, into: destination) { net, url in
            try NFKMLXSAM.loadWeights(into: net as! NFKMLXSAMNet, from: url)
        }
    }

    func testColorizerRoundTripsThroughItsOwnLoader() throws {
        try requireMLXRuntime()
        try assertRoundTrips(NFKMLXColorizer.makeNet(), into: NFKMLXColorizer.makeNet()) { net, url in
            try NFKMLXColorizer.loadWeights(into: net as! NFKMLXColorizerNet, from: url)
        }
    }

    func testDemucsRoundTripsThroughItsOwnLoader() throws {
        try requireMLXRuntime()
        try assertRoundTrips(NFKMLXDemucs.makeNet(), into: NFKMLXDemucs.makeNet()) { net, url in
            try NFKMLXDemucs.loadWeights(into: net as! NFKMLXDemucsNet, from: url)
        }
    }

    // Silero VAD carries 3-D Conv1d weights (the STFT basis and the encoder), so a fine-tuned save must
    // skip the PyTorch transpose the released `.jit` needs. The LSTM parameters round-trip as the
    // module's own `Wx`/`Wh`/`bias` rather than through the reference bias fold.
    func testSileroVADRoundTripsThroughItsOwnLoader() throws {
        try requireMLXRuntime()
        try assertRoundTrips(NFKMLXSileroVAD.makeNet(), into: NFKMLXSileroVAD.makeNet()) { net, url in
            try NFKMLXSileroVAD.loadWeights(into: net as! NFKMLXSileroVADNet, from: url)
        }
    }

    // DAC's decoder carries weight-normalized transposed convolutions (the shared vocoder block's
    // `conv_t1`) and Snake `alpha`; a fine-tuned save is already fused and in the module's layout, so the
    // loader must skip both the weight-norm fusion and every transpose. The codebook embeddings ride
    // along as ordinary 2-D parameters.
    func testDACRoundTripsThroughItsOwnLoader() throws {
        try requireMLXRuntime()
        try assertRoundTrips(NFKMLXDAC.makeNet(.tiny), into: NFKMLXDAC.makeNet(.tiny)) { net, url in
            try NFKMLXDAC.loadWeights(into: net as! NFKMLXDACNet, from: url)
        }
    }

    // SNAC carries depthwise convolutions, weight-normalized transposed convolutions, and the decoder
    // noise block's 1×1 conv; a fine-tuned save is fused and in the module's layout, so the loader skips
    // both the parametrization fusion and every transpose.
    func testSNACRoundTripsThroughItsOwnLoader() throws {
        try requireMLXRuntime()
        try assertRoundTrips(NFKMLXSNAC.makeNet(.tiny), into: NFKMLXSNAC.makeNet(.tiny)) { net, url in
            try NFKMLXSNAC.loadWeights(into: net as! NFKMLXSNACNet, from: url)
        }
    }

    // SigLIP 2's only transposed weight is the 4-D patch convolution; a fine-tuned save is in the
    // module's layout, so the loader must skip that transpose. The probe and fused attention projection
    // are 2-D/3-D and ride along untouched.
    func testSigLIP2RoundTripsThroughItsOwnLoader() throws {
        try requireMLXRuntime()
        try assertRoundTrips(NFKMLXSigLIP2.makeNet(.tiny), into: NFKMLXSigLIP2.makeNet(.tiny)) { net, url in
            try NFKMLXSigLIP2.loadWeights(into: net as! NFKMLXSigLIP2Net, from: url)
        }
    }

    // A saved music vocoder is already fused and in the module's layout, so the loader must skip both
    // the weight-norm fusion and every transpose — the alpha reshape included, which is this model's
    // own variant of the double-transpose hazard.
    func testMusic3VocoderRoundTripsThroughItsOwnLoader() throws {
        try requireMLXRuntime()
        try assertRoundTrips(NFKMLXMusic3.makeVocoder(.tiny), into: NFKMLXMusic3.makeVocoder(.tiny)) { net, url in
            try NFKMLXMusic3.loadVocoderWeights(into: net as! NFKMusic3VocoderNet, from: url)
        }
    }

    func testMusic3DepthDecoderRoundTripsThroughItsOwnLoader() throws {
        try requireMLXRuntime()
        try assertRoundTrips(NFKMLXMusic3.makeDepthDecoder(.tiny),
                             into: NFKMLXMusic3.makeDepthDecoder(.tiny)) { net, url in
            try NFKMLXMusic3.loadDepthWeights(into: net as! NFKMusic3DepthDecoderNet, from: url)
        }
    }

    func testMusic3DiTRoundTripsThroughItsOwnLoader() throws {
        try requireMLXRuntime()
        try assertRoundTrips(NFKMLXMusic3.makeDiT(.tiny), into: NFKMLXMusic3.makeDiT(.tiny)) { net, url in
            try NFKMLXMusic3.loadDiTWeights(into: net as! NFKMusic3DiTNet, from: url)
        }
    }

    func testMusic3ConditionEncoderRoundTripsThroughItsOwnLoader() throws {
        try requireMLXRuntime()
        try assertRoundTrips(NFKMLXMusic3.makeConditionEncoder(.tiny),
                             into: NFKMLXMusic3.makeConditionEncoder(.tiny)) { net, url in
            try NFKMLXMusic3.loadConditionWeights(into: net as! NFKMusic3ConditionEncoderNet, from: url)
        }
    }
}
