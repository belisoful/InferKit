//
//  NFKMLXPhonemizerTests.swift
//  InferKitMLXTests
//
//  Grapheme mapping and espeak availability need no GPU. The neural G2P model evaluates MLX arrays,
//  so those skip under `swift test` and run under `xcodebuild test`.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXPhonemizerTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func tinyConfiguration() -> NFKMLXG2PConfiguration {
        var configuration = NFKMLXG2PConfiguration()
        configuration.graphemeVocab = 40
        configuration.phonemeVocab = 20
        configuration.state = 32
        configuration.heads = 2
        configuration.encoderLayers = 1
        configuration.decoderLayers = 1
        configuration.maxLength = 8
        return configuration
    }

    // MARK: Grapheme mapping (no MLX)

    func testGraphemeMapping() {
        XCTAssertEqual(NFKMLXNeuralG2P.graphemeID("a"), 1)
        XCTAssertEqual(NFKMLXNeuralG2P.graphemeID("z"), 26)
        XCTAssertEqual(NFKMLXNeuralG2P.graphemeID(" "), 27)
        XCTAssertEqual(NFKMLXNeuralG2P.graphemeID("!"), 0)
    }

    // MARK: espeak path (no MLX; skips when not installed)

    #if os(macOS)
    func testEspeakPhonemizesWhenInstalled() throws {
        guard let espeak = NFKMLXEspeakPhonemizer() else {
            throw XCTSkip("espeak-ng not installed; run Tools/espeak/install.sh")
        }
        XCTAssertFalse(espeak.phonemes(for: "hello world").isEmpty, "espeak returns phoneme symbols")
    }
    #endif

    // MARK: Neural G2P (needs MLX)

    func testNeuralG2PProducesPhonemesWithinTheLimit() throws {
        try requireMLXRuntime()
        let symbols = (0 ..< 20).map { "p\($0)" }
        let g2p = NFKMLXNeuralG2P(configuration: tinyConfiguration(), phonemeSymbols: symbols)
        let phonemes = g2p.phonemes(for: "cat")
        XCTAssertLessThanOrEqual(phonemes.count, tinyConfiguration().maxLength)
        for symbol in phonemes { XCTAssertTrue(symbols.contains(symbol), "\(symbol) is a known phoneme symbol") }
    }

    func testNeuralG2PCheckpointRoundTripReproducesTheOutput() throws {
        try requireMLXRuntime()
        let trained = NFKMLXNeuralG2P(configuration: tinyConfiguration())
        let flattened = trained.model.parameters().flattened()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("g2p-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: Dictionary(uniqueKeysWithValues: flattened), url: url)

        let loaded = NFKMLXNeuralG2P(configuration: tinyConfiguration())
        try loaded.loadWeights(from: url)

        let graphemes = "cat".lowercased().unicodeScalars.map(NFKMLXNeuralG2P.graphemeID)
        XCTAssertEqual(trained.model.phonemeIDs(for: graphemes), loaded.model.phonemeIDs(for: graphemes))
    }
}
