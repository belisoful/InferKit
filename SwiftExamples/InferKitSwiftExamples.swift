//
//  InferKitSwiftExamples.swift
//  InferKitSwiftExamples
//
//  The Swift half of the compiled examples. `Examples/NFKExamples.m` is the Objective-C half; both
//  mirror Docs/examples.md, so CI catches a documented snippet that stops compiling. This file also
//  pins the shape the API takes once Swift's ObjC importer has renamed things — `runInference(for:)`
//  rather than `runInferenceForRequest:error:`, throwing rather than an `NSError **` out-parameter —
//  which is what a Swift consumer actually writes.
//
//  The weight-free paths run and assert; the model and network backends are exercised at the contract
//  level (construct, read identity), so their example code keeps compiling without weights or a server.
//

import XCTest
import CoreML
import InferKit

final class InferKitSwiftExamples: XCTestCase {

    // MARK: The shared contract (Docs/examples.md: The shared contract)

    func testExampleTheSharedContractWithPassthrough() throws {
        let backend = NFKPassthroughBackend()
        backend.outputMap = [NFKOutputText: NFKInputPrompt]   // each output key maps to an input key

        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "Explain diffraction in one sentence."],
                                          parameters: [NFKParameterMaxTokens: 64],
                                          outputModality: .text)

        // The ObjC `NSError **` out-parameter imports as `throws`.
        let result = try backend.runInference(for: request)
        XCTAssertEqual(result.text, "Explain diffraction in one sentence.")
    }

    // MARK: Jobs, submit, completion (Docs/examples.md: Subsystems — Jobs)

    func testExampleSubmittingAJob() {
        let backend = NFKPassthroughBackend()
        backend.outputMap = [NFKOutputImage: NFKInputImage]
        let request = NFKInferenceRequest(inputs: [NFKInputImage: "plate"])

        let finished = expectation(description: "job finished")
        let job = NFKInferenceSubmit(backend, request, nil)
        job.completionHandler = { completed in
            XCTAssertEqual(completed.status, .succeeded)
            XCTAssertEqual(completed.result?.output(forKey: NFKOutputImage) as? String, "plate")
            finished.fulfill()
        }
        wait(for: [finished], timeout: 5)
    }

    // MARK: Tensor conversion (Docs/examples.md: Subsystems — Tensor conversion)

    func testExampleTensorConversionRoundTrip() {
        var interleavedRGBA: [Float] = [
            0.1, 0.2, 0.3, 1.0,   0.4, 0.5, 0.6, 1.0,
            0.7, 0.8, 0.9, 1.0,   0.15, 0.25, 0.35, 1.0,
        ]
        let spec = NFKTensorSpecMake(2, 2, 3)                 // defaults: CHW, RGBA order, mean 0 / scale 1

        var tensor = [Float](repeating: 0, count: Int(NFKTensorElementCount(spec)))
        NFKInterleavedToTensor(&interleavedRGBA, &tensor, spec)
        XCTAssertEqual(NFKTensorElementCount(spec), 12)

        var restored = [Float](repeating: 0, count: 16)
        NFKTensorToInterleaved(&tensor, &restored, spec)
        XCTAssertEqual(restored[0], 0.1, accuracy: 1e-5)
    }

    // MARK: MLMultiArray bridge (Docs/examples.md: Subsystems — Tensor conversion)

    func testExampleMLMultiArrayBridge() throws {
        var interleavedRGBA: [Float] = [
            0.1, 0.2, 0.3, 1.0,   0.4, 0.5, 0.6, 1.0,
            0.7, 0.8, 0.9, 1.0,   0.15, 0.25, 0.35, 1.0,
        ]
        let spec = NFKTensorSpecMake(2, 2, 3)

        let array = try XCTUnwrap(NFKMultiArrayFromInterleaved(&interleavedRGBA, spec, nil))
        XCTAssertEqual(array.shape, [1, 3, 2, 2])             // [1, C, H, W]

        var restored = [Float](repeating: 0, count: 16)
        XCTAssertTrue(NFKInterleavedFromMultiArray(array, &restored, spec))
        XCTAssertEqual(restored[0], 0.1, accuracy: 1e-5)
    }

    // MARK: Tokenizer (Docs/examples.md: Subsystems — Tokenizers)

    func testExampleTokenizerEncodeDecode() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let vocabulary = ["h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hell": 6, "hello": 7]
        try JSONSerialization.data(withJSONObject: vocabulary)
            .write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\nh e\nl l\nhe ll\nhell o\n"
            .write(to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)

        let manifest = ["tokenizer": ["type": "bpe-bytelevel", "vocab": "vocab.json", "merges": "merges.txt"]]
        // A `nullable instancetype` factory imports as a throwing initializer.
        let tokenizer = try NFKTokenizer(forManifest: manifest, directory: directory)
        XCTAssertEqual(tokenizer.encode("hello"), [7])
        XCTAssertEqual(tokenizer.decode([7]), "hello")
    }

    // Docs/examples.md: Subsystems — Tokenizers. The CLIP variant, which the Stable Diffusion text
    // encoders take: text lowercases, and a word's last piece carries "</w>".
    func testExampleCLIPTokenizerEncodeDecode() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let vocabulary = ["h": 0, "e": 1, "l": 2, "o</w>": 3, "he": 4, "ll": 5, "hell": 6, "hello</w>": 7]
        try JSONSerialization.data(withJSONObject: vocabulary)
            .write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\nh e\nl l\nhe ll\nhell o</w>\n"
            .write(to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)

        let manifest = ["tokenizer": ["type": "clip", "vocab": "vocab.json", "merges": "merges.txt"]]
        let tokenizer = try NFKTokenizer(forManifest: manifest, directory: directory)
        XCTAssertEqual(tokenizer.encode("HELLO"), [7])              // lowercased first
        XCTAssertEqual(tokenizer.decode([7]), "hello")
    }

    // MARK: Hugging Face hub (Docs/examples.md: Subsystems — Hugging Face hub)

    func testExampleHuggingFaceHubURLResolution() {
        let hub = NFKHFHub(cacheDirectoryURL: nil)
        // The URL is optional: an invalid repo or path resolves to nil rather than a bad request.
        let remote = hub.remoteURL(forRepo: "Qwen/Qwen2.5-0.5B-Instruct", revision: nil,
                                   path: "tokenizer.json")
        XCTAssertEqual(remote?.absoluteString.contains("Qwen/Qwen2.5-0.5B-Instruct"), true)
    }

    // MARK: Backend contracts (Docs/examples.md: Text → text, Image → image)

    func testExampleLocalLanguageBackendContract() throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, *) else {
            throw XCTSkip("the Core ML language backend needs macOS 15 / iOS 18")
        }
        let backend = NFKCoreMLLanguageBackend(modelDirectoryURL: URL(fileURLWithPath: "/models/qwen"))
        backend.computeUnits = .all
        XCTAssertEqual(backend.backendIdentifier, "coreml-llm")
        XCTAssertFalse(backend.isReady)                       // not ready until prepared
    }

    func testExampleRemoteBackendContract() {
        let backend = NFKRemoteBackend(endpointURL: URL(string: "http://localhost:11434/v1/chat/completions"))
        backend.modelName = "llama3.2"
        XCTAssertEqual(backend.backendIdentifier, "remote")
        XCTAssertTrue(backend.isReady)                        // an endpoint is set
    }

    func testExampleTranscriptionBackendContract() {
        // Audio → text: point at an OpenAI-compatible transcriptions endpoint, set the model, and pass
        // audio under `NFKInputAudio` (an `NFKAudioAsset` or `Data`). The call itself needs the network.
        let backend = NFKRemoteTranscriptionBackend(
            endpointURL: URL(string: "https://api.example.com/v1/audio/transcriptions"))
        backend.modelName = "whisper-1"
        XCTAssertEqual(backend.backendIdentifier, "remote-transcription")
        XCTAssertTrue(backend.isReady)
    }

    func testExampleCoreMLImageBackendContract() {
        let backend = NFKCoreMLBackend(modelURL: URL(fileURLWithPath: "/models/style.mlpackage"))
        XCTAssertEqual(backend.backendIdentifier, "coreml")
        XCTAssertFalse(backend.isReady)
    }

    // MARK: Dynamic discovery (Docs/examples.md: Choosing a backend at runtime)

    func testExampleACapabilityIsUnavailableWithoutItsCompanion() {
        // The core resolves a provider class by name, so an engine that is not linked is simply
        // absent rather than a link error. This target links only the core.
        XCTAssertFalse(NFKDynamicBackend.isCapabilityAvailable(NFKCapabilityControlNet))
        XCTAssertThrowsError(try NFKDynamicBackend.backend(forCapability: NFKCapabilityControlNet))
    }

    // Remote providers from Swift: the same presets, with the importer's own naming.
    func testRemoteProviderPresets() {
        let ollama = NFKRemoteProvider(identifier: "ollama")
        XCTAssertNotNil(ollama)
        XCTAssertFalse(ollama!.requiresAPIKey, "a local server needs no key")

        let backend = NFKRemoteProvider.backend(for: ollama!, apiKey: nil, modelName: "llama3.2")
        XCTAssertTrue(backend.isReady)

        // Anthropic is the one provider that is not OpenAI-compatible.
        let claude = NFKRemoteProvider.backend(for: .anthropic, apiKey: "sk-ant-…",
                                               modelName: "claude-sonnet-4-5")
        XCTAssertEqual(claude.backendIdentifier, "anthropic-messages")
    }

    // MARK: Where Core ML runs (Docs/examples.md: Where Core ML actually runs)

    // `MLComputeUnitsCPUOnly` is zero, so an unset property would move every model off the
    // accelerators. The backend initializes it explicitly.
    func testExampleTheCoreMLBackendDefaultsToAllComputeUnits() {
        let backend = NFKCoreMLBackend(modelURL: nil)
        XCTAssertEqual(backend.computeUnits, .all)

        backend.computeUnits = .cpuAndNeuralEngine
        XCTAssertEqual(backend.computeUnits, .cpuAndNeuralEngine)
    }

    // The plan reads a compiled model's placement without running it. This pins the shape the ObjC
    // importer gives the factory, and the availability answer, without needing a model.
    func testExampleTheComputePlanReportsWhetherItCanAnswer() throws {
        if #available(macOS 14.4, iOS 17.4, tvOS 17.4, *) {
            XCTAssertTrue(NFKComputePlan.isAvailable)
        } else {
            XCTAssertFalse(NFKComputePlan.isAvailable)
        }

        // A model that is not there fails rather than reporting an empty plan, because "nothing is on
        // the Neural Engine" and "cannot tell" are different answers.
        let absent = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("inferkit-example-absent.mlmodelc")
        XCTAssertThrowsError(try NFKComputePlan(forCompiledModelAt: absent, computeUnits: .all))
    }
}
