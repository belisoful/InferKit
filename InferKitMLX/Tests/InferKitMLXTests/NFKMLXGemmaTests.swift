//
//  NFKMLXGemmaTests.swift
//  InferKitMLXTests
//
//  The Gemma 4 text decoder. No weights are loaded and no numerics are measured: Gemma 4 is in no
//  released `transformers` (4.57.6 is the newest on PyPI and stops at gemma3n), and the version that
//  supports it needs Python 3.10 while this machine has 3.9.6, so there is no oracle to compare
//  against. The verification is structural — and exact, because this checkpoint is bf16 and a float
//  module's parameters correspond one-to-one with it.
//

import XCTest
import MLX
@testable import InferKitMLX

final class NFKMLXGemmaTests: XCTestCase {

    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private func released() throws -> (NFKMLXGemmaConfiguration, [String: [Int]]) {
        guard let shapesPath = config["IK_SHAPES_GEMMA4"], let configPath = config["IK_CONFIG_GEMMA4"],
              let data = FileManager.default.contents(atPath: shapesPath),
              let shapes = try JSONSerialization.jsonObject(with: data) as? [String: [Int]]
        else { throw XCTSkip("set IK_SHAPES_GEMMA4 and IK_CONFIG_GEMMA4") }
        return (try NFKMLXGemmaLanguage.configuration(fromHuggingFace: URL(fileURLWithPath: configPath)),
                shapes)
    }

    private var small: NFKMLXGemmaConfiguration {
        NFKMLXGemmaConfiguration(hiddenSize: 64, layerCount: 4, intermediateSize: 128,
                                 vocabularySize: 128, headCount: 4, keyValueHeadCount: 1,
                                 headDimensions: 16, slidingWindow: 4, perLayerInputSize: 8,
                                 sharedKeyValueLayers: 2, finalLogitSoftcap: 30)
    }

    // MARK: Configuration

    func testTheReleasedConfigurationIsRead() throws {
        let (geometry, _) = try released()
        XCTAssertEqual(geometry.hiddenSize, 1536)
        XCTAssertEqual(geometry.layerCount, 35)
        XCTAssertEqual(geometry.headCount, 8)
        XCTAssertEqual(geometry.keyValueHeadCount, 1)
        XCTAssertEqual(geometry.headDimensions, 256)
        XCTAssertEqual(geometry.perLayerInputSize, 256)
        XCTAssertEqual(geometry.slidingWindow, 512)
        XCTAssertEqual(geometry.finalLogitSoftcap, 30)
    }

    // The wide second embedding holds one slice per layer, which is what its width has to be.
    /// The reader takes exactly the architecture it implements. `gemma4_unified_text` (the 12B) and
    /// the mixture-of-experts 26B both satisfy a `gemma4` PREFIX, which is how a wrong checkpoint
    /// would load cleanly and produce fluent nonsense — so the guard is exact, not a prefix.
    func testTheConfigurationRejectsTheArchitecturesItDoesNotImplement() throws {
        func write(_ json: [String: Any]) throws -> URL {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("gemma-guard-\(UUID().uuidString).json")
            try JSONSerialization.data(withJSONObject: json).write(to: url)
            return url
        }
        let unified = try write(["model_type": "gemma4_unified",
                                 "text_config": ["model_type": "gemma4_unified_text"]])
        defer { try? FileManager.default.removeItem(at: unified) }
        XCTAssertThrowsError(try NFKMLXGemmaLanguage.configuration(fromHuggingFace: unified))

        let mixture = try write(["model_type": "gemma4",
                                 "text_config": ["model_type": "gemma4_text", "num_experts": 128]])
        defer { try? FileManager.default.removeItem(at: mixture) }
        XCTAssertThrowsError(try NFKMLXGemmaLanguage.configuration(fromHuggingFace: mixture))

        let dense = try write(["model_type": "gemma4",
                               "text_config": ["model_type": "gemma4_text"]])
        defer { try? FileManager.default.removeItem(at: dense) }
        XCTAssertNoThrow(try NFKMLXGemmaLanguage.configuration(fromHuggingFace: dense))
    }

    func testThePerLayerEmbeddingIsOneSlicePerLayer() throws {
        let (geometry, shapes) = try released()
        XCTAssertEqual(geometry.perLayerEmbeddingWidth, 35 * 256)
        XCTAssertEqual(shapes["model.language_model.embed_tokens_per_layer.weight"],
                       [geometry.vocabularySize, geometry.perLayerEmbeddingWidth])
    }

    // MARK: Structure against the released checkpoint

    func testEveryParameterMatchesTheReleasedCheckpoint() throws {
        try requireMLXRuntime()
        let (geometry, shapes) = try released()
        let net = NFKMLXGemmaLanguage.makeNet(geometry)

        var checked = 0
        var missing = [String]()
        var mismatched = [String]()
        for (name, value) in net.parameters().flattened() {
            let key = NFKMLXGemmaLanguage.referenceKey(for: name)
            guard let expected = shapes[key] else { missing.append(key); continue }
            checked += 1
            if value.shape != expected {
                mismatched.append("\(key): built \(value.shape), released \(expected)")
            }
        }
        print("VALIDATION structure gemma4-e2b: \(checked) parameters checked, "
              + "\(missing.count) missing, \(mismatched.count) mismatched")
        XCTAssertTrue(mismatched.isEmpty, "shape mismatches:\n" + mismatched.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(missing.isEmpty, "declared but absent from the release:\n"
                      + missing.prefix(8).joined(separator: "\n"))
        XCTAssertGreaterThan(checked, 590, "the decoder's parameters are all accounted for")
    }

    // What the release carries that this does not: it is tri-modal.
    func testTheOtherTowersAreNamed() throws {
        let (_, shapes) = try released()
        let text = shapes.keys.filter { $0.hasPrefix("model.language_model.") }
        let vision = shapes.keys.filter { $0.hasPrefix("model.vision_tower.") || $0.hasPrefix("model.embed_vision.") }
        let audio = shapes.keys.filter { $0.hasPrefix("model.audio_tower.") || $0.hasPrefix("model.embed_audio.") }
        print("VALIDATION structure gemma4-e2b: text \(text.count), vision \(vision.count), audio \(audio.count)")
        XCTAssertGreaterThan(vision.count, 0, "the release carries a vision tower")
        XCTAssertGreaterThan(audio.count, 0, "and an audio Conformer")
        XCTAssertEqual(text.count + vision.count + audio.count, shapes.count,
                       "every tensor is the decoder's or a named tower's")
    }

    // MARK: Behavior

    // Gemma 4 normalizes with a PLAIN scale, `x · w`, and initializes the weight to one — unlike
    // Gemma 3, whose weight is an offset from one. Assuming the Gemma 3 convention is what first made
    // this port wrong, so the difference is pinned here rather than left to memory.
    func testTheNormalizationIsAPlainScale() throws {
        try requireMLXRuntime()
        let norm = NFKGemmaNorm(dimensions: 8, eps: 1e-6)
        let x = MLXArray((0 ..< 8).map { Float($0 + 1) }).reshaped([1, 8])
        let out = norm(x)
        eval(out)

        // With a weight of one the result is the RMS-normalized input itself.
        let values = x.reshaped([-1]).asArray(Float.self)
        let meanSquare = values.reduce(0) { $0 + $1 * $1 } / Float(values.count)
        let expected = values[7] / (meanSquare + 1e-6).squareRoot()
        XCTAssertEqual(Double(out[0, 7].item(Float.self)), Double(expected), accuracy: 1e-4)
    }

    func testTheWindowMaskForbidsOlderPositions() {
        let mask = NFKMLXGemmaLanguage.windowMask(6, window: 3)
        eval(mask)
        let values = mask.asArray(Float.self)
        XCTAssertEqual(values[5 * 6 + 5], 0, "a position sees itself")
        XCTAssertEqual(values[5 * 6 + 3], 0, "and two back, inside the window")
        XCTAssertLessThan(values[5 * 6 + 2], -1e8, "but not three back, outside it")
        XCTAssertLessThan(values[0 * 6 + 1], -1e8, "and never the future")
    }

    func testASmallConfigurationRunsEndToEnd() throws {
        try requireMLXRuntime()
        let net = NFKMLXGemmaLanguage.makeNet(small)
        let logits = net(MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4]))
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 4, small.vocabularySize])
        // The soft cap bounds the logits by construction.
        let peak = logits.abs().max().item(Float.self)
        XCTAssertLessThanOrEqual(Double(peak), Double(small.finalLogitSoftcap) + 1e-3,
                                 "the final soft cap bounds every logit")
    }
}
