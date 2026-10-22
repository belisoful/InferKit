//
//  NFKMLXSmolVLMTests.swift
//  InferKitMLXTests
//
//  Weight-free structural tests for the SmolVLM2 vision-language model: the SigLIP encoder, the
//  pixel-shuffle connector, the image processor's tiling, and the prompt expansion. Numeric parity
//  against the reference lives in NFKMLXReferenceParityTests.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXSmolVLMTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        // Clearing the cache reaches MLX's runtime, which needs a Metal library it can find.
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    func testTheVisionEncoderProducesPatchFeatures() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(4)
        let net = NFKMLXSigLIPNet(.tiny)
        // Two tiles of a 64×64 image with 16-patch grid → 16 patches per tile.
        let pixels = MLXRandom.normal([2, 64, 64, 3])
        let features = net(pixels)
        XCTAssertEqual(features.shape, [2, 16, 32], "one feature per patch")
    }

    func testTheConnectorReducesTokensByScaleSquared() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(4)
        // A 8×8 = 64-patch grid at scale 4 folds to 64 / 16 = 4 tokens of 16× the channels projected.
        let connector = NFKMLXSmolVLMConnector(visionHidden: 32, decoderHidden: 48, scaleFactor: 4)
        let features = connector(MLXRandom.normal([2, 64, 32]))
        XCTAssertEqual(features.shape, [2, 4, 48], "scale² fewer tokens at the decoder width")
    }

    func testTheImageProcessorTilesToTheGrid() throws {
        try requireMLXRuntime()
        // A square image scales its longest edge to 2048 and splits 4×4, plus a global tile: 17 tiles.
        let square = NFKMLXSmolVLMTests.solidImage(width: 300, height: 300)
        let (pixels, rows, cols) = NFKMLXSmolVLMImageProcessor.process(square)
        XCTAssertEqual([rows, cols], [4, 4])
        XCTAssertEqual(pixels.shape, [17, 3, 512, 512], "16 sub-tiles and a global thumbnail")

        // A wide image scales to 2048×512 and splits 1×4, plus a global tile: 5 tiles.
        let wide = NFKMLXSmolVLMTests.solidImage(width: 1024, height: 256)
        let (widePixels, wideRows, wideCols) = NFKMLXSmolVLMImageProcessor.process(wide)
        XCTAssertEqual([wideRows, wideCols], [1, 4])
        XCTAssertEqual(widePixels.shape, [5, 3, 512, 512])

        // The pixels are normalized to -1 … 1.
        eval(pixels)
        let values = pixels.asArray(Float.self)
        XCTAssertGreaterThanOrEqual(values.min() ?? 0, -1.0001)
        XCTAssertLessThanOrEqual(values.max() ?? 0, 1.0001)
    }

    func testThePromptExpandsToTheRightImageTokenCount() {
        // A 2×2 tiling: 4 sub-tiles + 1 global, each 64 image tokens → 320 `<image>` markers.
        let prompt = NFKMLXSmolVLM.prompt(rows: 2, cols: 2, question: "What is this?")
        let count = prompt.components(separatedBy: "<image>").count - 1
        XCTAssertEqual(count, 5 * NFKMLXSmolVLM.tokensPerTile)
        XCTAssertTrue(prompt.hasPrefix("User:"))
        XCTAssertTrue(prompt.contains("<row_2_col_2>"))
        XCTAssertTrue(prompt.contains("<global-img>"))
        XCTAssertTrue(prompt.hasSuffix("<end_of_utterance>\nAssistant:"))
    }

    func testFusionSplicesVisionTokensAtImagePositions() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(4)
        let vision = NFKMLXSigLIPNet(.tiny)
        let connector = NFKMLXSmolVLMConnector(visionHidden: 32, decoderHidden: 64, scaleFactor: 4)
        var decoderConfiguration = NFKMLXLanguageConfiguration.tiny
        decoderConfiguration.hiddenSize = 64
        let decoder = NFKMLXLanguage.makeNet(decoderConfiguration)
        let net = NFKMLXSmolVLMNet(vision: vision, connector: connector, decoder: decoder, imageTokenId: 7)

        // Two image tokens (id 7) among text tokens; two vision features to splice in.
        let features = MLXRandom.normal([1, 2, 64])
        let embeddings = net.fusedEmbeddings(inputIds: [3, 7, 7, 5], imageFeatures: features)
        eval(embeddings)
        XCTAssertEqual(embeddings.shape, [1, 4, 64])
        // The image positions carry the (flattened) features, not the token embedding.
        let spliced = embeddings[0, 1].asArray(Float.self)
        let expected = features.reshaped([-1, 64])[0].asArray(Float.self)
        XCTAssertEqual(spliced, expected, "an image position reads its vision feature")
    }

    private static func solidImage(width: Int, height: Int) -> CGImage {
        var bytes = [UInt8](repeating: 128, count: width * height * 4)
        let context = CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        return context.makeImage()!
    }

    // A release whose tokenizer.json is missing is refused when it is built; it would otherwise answer
    // every question with an empty string.
    func testAReleaseWithoutATokenizerIsRefused() throws {
        try requireMLXRuntime()
        guard let store = NFKMLXValidationConfig.environment["IK_VAL_SMOLVLM2_256M"] else {
            throw XCTSkip("set IK_VAL_SMOLVLM2_256M")
        }
        let partial = try release(store, without: ["tokenizer.json"])
        defer { try? FileManager.default.removeItem(at: partial) }
        XCTAssertThrowsError(try NFKMLXSmolVLM.load(directoryURL: partial)) { error in
            XCTAssertTrue("\(error)".contains("no readable tokenizer"), "\(error)")
        }
    }

    // The ids are transformers' `AutoTokenizer` on each release (`run_reference.py smolvlm_tokenizer`),
    // as the processor calls it. Each whitespace run ending at a number encodes as one token, where
    // GPT-2's pattern leaves the run's last space to encode on its own; the oracle checks that every
    // case encodes differently under GPT-2's pattern. The image markers and `<end_of_utterance>` encode
    // as their added tokens.
    static let digitCases: [String] = [
        "User:Set x =   42 and y =\t\t7, then add  12.<end_of_utterance>\nAssistant:",
        "User:<fake_token_around_image><row_1_col_2><image><image>\n<fake_token_around_image><global-img><image>"
            + "<fake_token_around_image>Read 1,299.99 and   007 aloud.\n\n3<end_of_utterance>\nAssistant: 42",
    ]

    // The three releases share one vocabulary and encode every case to the same ids.
    static let digitIds: [[Int]] = [
        [11126, 42, 7143, 1792, 446, 333, 36, 34, 284, 329, 446, 1656, 39, 28, 965, 803, 256, 33, 34, 30, 49279, 198,
         9519, 9531, 42],
        [11126, 42, 49189, 49154, 49190, 49190, 198, 49189, 49152, 49190, 49189, 6340, 216, 33, 28, 34, 41, 41, 30, 41,
         41, 284, 333, 32, 32, 39, 16076, 30, 1116, 35, 49279, 198, 9519, 9531, 42, 216, 36, 34],
    ]

    static let digitReleases = ["IK_VAL_SMOLVLM2_256M", "IK_VAL_SMOLVLM2", "IK_VAL_SMOLVLM2_2_2B"]

    func testADigitBearingPromptEncodesAsTransformersDoes() throws {
        var checked = [String]()
        for key in Self.digitReleases {
            guard let path = NFKMLXValidationConfig.environment[key] else { continue }
            let tokenizer = try NFKMLXSmolVLM.tokenizer(inDirectory: URL(fileURLWithPath: path))
            for (text, expected) in zip(Self.digitCases, Self.digitIds) {
                XCTAssertEqual(tokenizer.encode(text).map(\.intValue), expected, "\(key): \(text.debugDescription)")
            }
            checked.append(key)
        }
        try XCTSkipIf(checked.isEmpty, "set any of " + Self.digitReleases.joined(separator: ", "))
    }

    // A release whose pre-tokenizer the reader does not reproduce is refused. SmolLM's scheme with the
    // digits kept together is no longer SmolLM's.
    func testATokenizerSchemeTheReaderDoesNotReproduceIsRefused() throws {
        guard let store = NFKMLXValidationConfig.environment["IK_VAL_SMOLVLM2_256M"] else {
            throw XCTSkip("set IK_VAL_SMOLVLM2_256M")
        }
        let source = URL(fileURLWithPath: store).appendingPathComponent("tokenizer.json")
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: source)) as? [String: Any])
        var preTokenizer = try XCTUnwrap(json["pre_tokenizer"] as? [String: Any])
        var steps = try XCTUnwrap(preTokenizer["pretokenizers"] as? [[String: Any]])
        steps[0]["individual_digits"] = false
        preTokenizer["pretokenizers"] = steps
        json["pre_tokenizer"] = preTokenizer
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONSerialization.data(withJSONObject: json).write(to: directory.appendingPathComponent("tokenizer.json"))
        XCTAssertThrowsError(try NFKMLXSmolVLM.tokenizer(inDirectory: directory)) { error in
            XCTAssertTrue("\(error)".contains("pre-tokenizer"), "\(error)")
        }
    }

    /// A copy of a release directory with some files left out, linked file by file.
    private func release(_ store: String, without omitted: Set<String>) throws -> URL {
        let source = URL(fileURLWithPath: store)
        let copy = FileManager.default.temporaryDirectory.appendingPathComponent("partial-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: copy, withIntermediateDirectories: true)
        for name in try FileManager.default.contentsOfDirectory(atPath: source.path) where !omitted.contains(name) {
            try FileManager.default.createSymbolicLink(at: copy.appendingPathComponent(name),
                                                       withDestinationURL: source.appendingPathComponent(name))
        }
        return copy
    }

}
