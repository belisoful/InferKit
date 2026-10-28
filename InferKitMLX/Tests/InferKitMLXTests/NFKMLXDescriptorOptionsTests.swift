//
//  NFKMLXDescriptorOptionsTests.swift
//  InferKitMLXTests
//
//  The embedding width and Canary's languages a request sets, and the catalog options that declare them.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXDescriptorOptionsTests: XCTestCase {

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
    }

    private func request(_ parameters: [String: Any]) -> NFKInferenceRequest {
        NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"], parameters: parameters)
    }

    // MARK: Embedding width

    func testTheEmbeddingWidthTruncatesAndNormalizesAgain() throws {
        try requireMLXRuntime()
        let full = MLXArray([0.6, 0.0, 0.8, 0.0] as [Float])
        let narrowed = try NFKMLXTextEmbeddingBackend.embedding(full, for: request([NFKParameterEmbeddingDimensions: 1]))
        XCTAssertEqual(narrowed.asArray(Float.self), [1])
        let two = try NFKMLXTextEmbeddingBackend.embedding(full, for: request([NFKParameterEmbeddingDimensions: 2]))
        XCTAssertEqual(two.asArray(Float.self), [1, 0])
        let three = try NFKMLXTextEmbeddingBackend.embedding(full, for: request([NFKParameterEmbeddingDimensions: 3]))
        XCTAssertEqual(three.asArray(Float.self), [0.6, 0, 0.8], "already unit length")
        XCTAssertEqual(try NFKMLXTextEmbeddingBackend.embedding(full, for: request([NFKParameterEmbeddingDimensions: 9]))
                        .asArray(Float.self), full.asArray(Float.self), "a width past the embedding's returns it whole")
        XCTAssertEqual(try NFKMLXTextEmbeddingBackend.embedding(full, for: request([:])).asArray(Float.self),
                       full.asArray(Float.self))
        XCTAssertThrowsError(try NFKMLXTextEmbeddingBackend.embedding(full, for: request([NFKParameterEmbeddingDimensions: 0])))
    }

    func testATextEmbeddingBackendDeclaresTheWidth() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXQwen3Embedding.backend(weightsURL: nil, tokenizer: nil, configuration: .tiny)
        XCTAssertEqual(backend.supportedParameterKeys, [NFKParameterEmbeddingDimensions])
        XCTAssertEqual(NFKParameterEmbeddingDimensions, "dimensions", "the OpenAI embeddings field, sent as is")
    }

    // A request's width equals the factory's: the embedding truncated, then normalized again.
    func testARequestWidthEqualsTheFactoryWidth() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        let releases: [(key: String, full: (URL) throws -> any NFKInferenceBackend, narrow: (URL) throws -> any NFKInferenceBackend)] = [
            ("IK_VAL_QWEN3_EMBEDDING", { try NFKMLXQwen3Embedding.backend(directoryURL: $0) },
             { try NFKMLXQwen3Embedding.backend(directoryURL: $0, outputDimensions: 256) }),
            ("IK_VAL_EMBEDDINGGEMMA", { try NFKMLXEmbeddingGemma.backend(directoryURL: $0) },
             { try NFKMLXEmbeddingGemma.backend(directoryURL: $0, outputDimensions: 256) }),
        ]
        var ran = 0
        for release in releases {
            guard let directory = env[release.key], FileManager.default.fileExists(atPath: directory) else { continue }
            let url = URL(fileURLWithPath: directory)
            func vector(_ backend: any NFKInferenceBackend, _ parameters: [String: Any]) throws -> [Float] {
                let result = try backend.runInference(for: NFKInferenceRequest(
                    inputs: [NFKInputPrompt: "The quick brown fox jumps over the lazy dog."], parameters: parameters))
                return try XCTUnwrap(result.output(forKey: NFKOutputEmbedding) as? [NSNumber]).map(\.floatValue)
            }
            let byRequest = try vector(try release.full(url), [NFKParameterEmbeddingDimensions: 256])
            NFKMLXGPU.clearCache()
            let byFactory = try vector(try release.narrow(url), [:])
            NFKMLXGPU.clearCache()
            XCTAssertEqual(byRequest.count, 256, release.key)
            XCTAssertEqual(byFactory.count, 256, release.key)
            let difference = zip(byRequest, byFactory).map { abs($0 - $1) }.max() ?? 1
            print("VALIDATION \(release.key) width 256: request against factory max |Δ| \(difference)")
            XCTAssertLessThan(difference, 1e-5, release.key)
            ran += 1
        }
        if ran == 0 {
            throw XCTSkip("set IK_VAL_QWEN3_EMBEDDING or IK_VAL_EMBEDDINGGEMMA")
        }
    }

    // MARK: Canary

    func testCanaryReadsItsLanguagesFromOptions() throws {
        let source = NFKParameterSourceLanguage, target = NFKParameterTargetLanguage
        XCTAssertTrue(try NFKMLXCanaryBackend.languages(from: request([:])) == ("en", "en"))
        XCTAssertTrue(try NFKMLXCanaryBackend.languages(from: request([source: "de"])) == ("de", "de"))
        XCTAssertTrue(try NFKMLXCanaryBackend.languages(from: request([source: "de", target: "en"])) == ("de", "en"))
        XCTAssertTrue(try NFKMLXCanaryBackend.languages(from: request([target: "FR"])) == ("en", "fr"))
        XCTAssertThrowsError(try NFKMLXCanaryBackend.languages(from: request([source: "ja"])), "not a Canary language")
        XCTAssertThrowsError(try NFKMLXCanaryBackend.languages(from: request([source: "de", target: "fr"])),
                             "a translation has English on one side")
        XCTAssertThrowsError(try NFKMLXCanaryBackend.languages(from: request([source: 3])))
        XCTAssertEqual(NFKMLXCanary.supportedLanguages.count, 25)
    }

    // MARK: Catalog

    private func descriptor(_ name: String) throws -> NFKModelDescriptor {
        try XCTUnwrap(NFKMLXModelDescriptors.modelDescriptors().first { $0.identifier == name }, name)
    }

    func testTheCatalogDeclaresTheWidthsAndTheLanguages() throws {
        let gemma = try XCTUnwrap(try descriptor(NFKMLXEmbeddingGemma.modelName).option(withKey: NFKParameterEmbeddingDimensions))
        XCTAssertEqual(gemma.choices.map { $0.value as? Int }, [768, 512, 256, 128])
        let qwen = try XCTUnwrap(try descriptor(NFKMLXQwen3Embedding.modelName).option(withKey: NFKParameterEmbeddingDimensions))
        XCTAssertEqual(qwen.minimum, 32)
        XCTAssertEqual(qwen.maximum, 1024)
        XCTAssertEqual(qwen.defaultValue as? Int, 1024)
        let vision = try XCTUnwrap(try descriptor(NFKMLXQwen3VLEmbedder.modelName).option(withKey: NFKParameterEmbeddingDimensions))
        XCTAssertEqual(vision.minimum, 64)
        XCTAssertEqual(vision.maximum, 2048)

        let canary = try descriptor(NFKMLXCanary.modelName)
        XCTAssertEqual(Set(canary.inputs.map(\.key)), [NFKInputAudio], "the prompt no longer carries the languages")
        let language = try XCTUnwrap(canary.option(withKey: NFKParameterSourceLanguage))
        XCTAssertEqual(language.defaultValue as? String, "en")
        XCTAssertEqual(language.choices.map { $0.value as? String }, NFKMLXCanary.supportedLanguages)
        XCTAssertEqual(language.choices.first { ($0.value as? String) == "sl" }?.title, "Slovenian (sl)")
        XCTAssertNil(canary.option(withKey: NFKParameterTargetLanguage)?.defaultValue, "no target transcribes")
    }
}
