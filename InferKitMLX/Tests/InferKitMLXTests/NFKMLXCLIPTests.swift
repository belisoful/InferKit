//
//  NFKMLXCLIPTests.swift
//  InferKitMLXTests
//
//  Parameter names match the reference OpenAI CLIP without a GPU. The forwards and the weight
//  round-trip evaluate MLX arrays, so they skip without a Metal library for MLX (see
//  Tools/mlx-metallib.sh).
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXCLIPTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func tinyNet() -> NFKMLXCLIPNet {
        NFKMLXCLIPNet(.tiny)
    }

    // MARK: Parameter names

    func testParameterNamesMatchTheReferenceCheckpoint() throws {
        try requireMLXRuntime()
        let names = Set(tinyNet().parameters().flattened().map(\.0))
        for expected in ["visual.conv1.weight", "visual.class_embedding", "visual.positional_embedding",
                         "visual.transformer.resblocks.0.attn.in_proj_weight",
                         "visual.transformer.resblocks.0.attn.out_proj.weight",
                         "visual.transformer.resblocks.0.mlp.c_fc.weight", "visual.proj",
                         "token_embedding.weight", "positional_embedding",
                         "transformer.resblocks.0.ln_1.weight", "ln_final.weight",
                         "text_projection", "logit_scale"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    // MARK: Forwards (needs MLX)

    func testImageEmbeddingIsUnitLengthAndDeterministic() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let image = Self.image(height: 20, width: 24)
        let a = net.encodeImage(image)
        let b = net.encodeImage(image)
        eval(a, b)
        XCTAssertEqual(a.shape, [NFKMLXCLIPConfiguration.tiny.embedDimensions])
        XCTAssertEqual(Self.norm(a), 1, accuracy: 1e-4, "L2-normalized")
        XCTAssertEqual(a.asArray(Float.self), b.asArray(Float.self), "deterministic")
    }

    func testTextEmbeddingIsUnitLengthInTheSharedSpace() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let embedding = net.encodeText([1, 8, 5, 63, 2])                    // 63 is the highest id → the EOT position
        eval(embedding)
        XCTAssertEqual(embedding.shape, [NFKMLXCLIPConfiguration.tiny.embedDimensions],
                       "text and image embeddings share a dimension")
        XCTAssertEqual(Self.norm(embedding), 1, accuracy: 1e-4, "L2-normalized")
    }

    func testASafetensorsCheckpointLoadsAndReproducesTheForward() throws {
        try requireMLXRuntime()
        let trained = tinyNet()

        // Save in PyTorch layout: only the patch-embedding convolution is 4-D and transposes.
        let pytorchLayout = Dictionary(uniqueKeysWithValues: trained.parameters().flattened().map { key, value in
            (key, value.ndim == 4 ? value.transposed(0, 3, 1, 2) : value)
        })
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("clip-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try save(arrays: pytorchLayout, url: url)

        let loaded = tinyNet()
        try NFKMLXCLIP.loadWeights(into: loaded, from: url)

        let image = Self.image(height: 16, width: 16)
        let expected = trained.encodeImage(image)
        let actual = loaded.encodeImage(image)
        eval(expected, actual)
        XCTAssertEqual(expected.asArray(Float.self), actual.asArray(Float.self),
                       "loaded weights reproduce the trained forward")
    }

    func testTheRegisteredBackendEmbedsACGImage() throws {
        try requireMLXRuntime()
        NFKMLXCLIP.register()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXCLIP.modelName))
        let backend = try NFKMLXModelRegistry.backend(named: NFKMLXCLIP.modelName, weightsURL: nil)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(8, 8)]))
        let embedding = try XCTUnwrap(result.embedding, "embedding output present")
        XCTAssertEqual(embedding.count, NFKMLXCLIPConfiguration.base.embedDimensions, "ViT-B/32 embedding width")
    }

    func testEachRegisteredReleaseEmbedsEveryInputItsCatalogEntryLists() throws {
        try requireMLXRuntime()
        NFKMLXCLIP.register()
        let values: [String: Any] = [NFKInputImage: Self.solid(8, 8), NFKInputPrompt: "a photo of a cat"]
        for variant in NFKMLXCatalogImage.clipReleases {
            let spec = NFKMLXCLIP.specs(for: variant)
            let entry = try XCTUnwrap(NFKMLXCatalogImage.clipEntries.first { $0.identifier == spec.name }, spec.name)
            let backend = try NFKMLXModelRegistry.backend(named: spec.name, weightsURL: nil)
            XCTAssertEqual(Set(entry.inputs.map(\.key)), backend.supportedInputKeys, spec.name)
            XCTAssertEqual(entry.inputs.first?.key, NFKInputImage, "\(spec.name) lists first the input every build reads")
            for port in entry.inputs {
                let value = try XCTUnwrap(values[port.key], "\(spec.name) lists \(port.key)")
                let result = try backend.runInference(for: NFKInferenceRequest(inputs: [port.key: value]))
                XCTAssertEqual(result.embedding?.count, spec.configuration.embedDimensions, "\(spec.name) \(port.key)")
            }
            NFKMLXGPU.clearCache()
        }
    }

    // MARK: Text through a tokenizer

    func testATokenizerDirectoryEmbedsAPromptInTheImagesSpace() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXCLIP.backend(variant: .vitB32, weightsURL: nil, tokenizerDirectoryURL: try tokenizerDirectory())
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputImage, NFKInputPrompt])
        let text = try XCTUnwrap(try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "a cat"])).embedding)
        let image = try XCTUnwrap(try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.solid(8, 8)])).embedding)
        XCTAssertEqual(text.count, image.count, "text and image embeddings share a dimension")
        XCTAssertEqual(sqrt(text.reduce(0) { $0 + $1.doubleValue * $1.doubleValue }), 1, accuracy: 1e-4, "L2-normalized")
    }

    func testTheTokenizerMarksAPromptAndPadsItToTheContextLength() throws {
        let tokenizer = try NFKMLXCLIP.tokenizer(in: try tokenizerDirectory())
        // a</w> c a t</w>, between the start marker (52) and the end marker (53), which also pads.
        XCTAssertEqual(tokenizer.tokens(for: "A  cat", contextLength: 8), [52, 26, 2, 0, 45, 53, 53, 53])
    }

    func testABuildWithoutATokenizerReadsAPromptOnlyOnInitialWeights() throws {
        let configuration = NFKMLXCLIPConfiguration.tiny
        let ids = NFKMLXCLIP.byteTokens("hi", configuration: configuration)
        let end = configuration.vocabularySize - 1
        XCTAssertEqual(ids.count, configuration.contextLength)
        XCTAssertEqual(Array(ids.prefix(4)), [end - 1, Int(UInt8(ascii: "h")) % (end - 1), Int(UInt8(ascii: "i")) % (end - 1), end])
        XCTAssertEqual(ids.firstIndex(of: ids.max() ?? 0), 3, "the text tower pools at the end marker")
        XCTAssertEqual(NFKMLXCLIP.byteTokens(String(repeating: "x", count: 100), configuration: configuration).count,
                       configuration.contextLength, "a long prompt is cut to the context length")
    }

    func testAReleaseDirectoryNamesWhatItLacks() throws {
        let directory = try tokenizerDirectory()
        XCTAssertThrowsError(try NFKMLXCLIP.backend(variant: .vitB32, directoryURL: directory)) { error in
            XCTAssertTrue("\(error)".contains("no CLIP checkpoint"), "\(error)")
        }
        let empty = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        XCTAssertThrowsError(try NFKMLXCLIP.tokenizer(in: empty)) { error in
            XCTAssertTrue("\(error)".contains("vocab.json"), "\(error)")
        }
    }

    // MARK: Helpers

    /// A CLIP tokenizer directory over 54 ids: each lowercase letter (0–25), the same letter ending a word
    /// (`a</w>` is 26), and the start and end markers (52 and 53), with no merges.
    private func tokenizerDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("clip-tokenizer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let letters = "abcdefghijklmnopqrstuvwxyz".map(String.init)
        var vocabulary = [String: Int]()
        for (index, letter) in letters.enumerated() {
            vocabulary[letter] = index
            vocabulary[letter + "</w>"] = letters.count + index
        }
        vocabulary["<|startoftext|>"] = 52
        vocabulary["<|endoftext|>"] = 53
        try JSONSerialization.data(withJSONObject: vocabulary).write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\n".write(to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        return directory
    }

    static func norm(_ x: MLXArray) -> Float {
        sqrtf(x.asArray(Float.self).reduce(0) { $0 + $1 * $1 })
    }

    static func image(height: Int, width: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: height * width * 3)
        for i in 0 ..< values.count {
            values[i] = Float((i * 37) % 256) / 255.0
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [height, width, 3]) }
    }

    static func solid(_ width: Int, _ height: Int) -> CGImage {
        let pixels = [UInt8](repeating: 128, count: width * height * 4)
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }
}
